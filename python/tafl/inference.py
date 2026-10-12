"""Batched GPU evaluation for engine pools, overlapping the GPU with tree search."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .model import Net
from .native import FEATURE_BYTES, NUM_ACTIONS, Pool


class Batch:
    """Pinned host buffers for one pool, so copies and the forward pass run asynchronously."""

    def __init__(self, rows: int, device: torch.device) -> None:
        pin = device.type == "cuda"
        self.planes = torch.empty((rows, FEATURE_BYTES), dtype=torch.uint8, pin_memory=pin)
        self.policy = torch.empty((rows, NUM_ACTIONS), dtype=torch.float32, pin_memory=pin)
        self.value = torch.empty(rows, dtype=torch.float32, pin_memory=pin)
        self.planes_np = self.planes.numpy()
        self.policy_np = self.policy.numpy()
        self.value_np = self.value.numpy()
        self.done = torch.cuda.Event() if pin else None
        self.rows = 0


@dataclass
class _Graph:
    graph: torch.cuda.CUDAGraph
    planes: torch.Tensor  # static input
    policy: torch.Tensor  # static outputs
    value: torch.Tensor


def graph_size(rows: int) -> int:
    """The batch size of the graph that serves `rows` rows: a power of two up to 128,
    then a multiple of 128 (so padding wastes at most half of a small batch and an
    eighth of a large one)."""
    return 1 << max(rows - 1, 0).bit_length() if rows <= 128 else -(-rows // 128) * 128


class Evaluator:
    """Evaluates leaves with `model`: a float32 network runs under fp16 autocast, a
    `Net.fused()` copy as it is.

    On a GPU the forward pass is replayed from CUDA graphs captured once per batch size
    (`graph_size`), which removes the cost of launching its kernels one by one from
    Python: about 10x faster for small batches when the CPU is busy. `set_model` then
    copies the new weights into the captured network, which must have the same shape.

    With `moves_left_alpha > 0` the value handed to the search is shaped by the
    predicted game length M: `v * (1 - alpha * min(M / moves_left_scale, 1))`, so a
    losing side prefers to hold out longer and a winning side to finish sooner. This
    gives the search a gradient when every outcome looks the same (as for the
    attackers early in training).
    """

    def __init__(
        self,
        model: Net,
        device: torch.device,
        moves_left_alpha: float = 0.0,
        moves_left_scale: float = 200.0,
        graphs: bool = True,
    ) -> None:
        self.model = model.eval()
        self.device = device
        self.moves_left_alpha = moves_left_alpha
        self.moves_left_scale = moves_left_scale
        self.evaluations = 0
        self.use_graphs = graphs and device.type == "cuda"
        self.graphs: dict[int, _Graph] = {}

    def value(self, out: dict[str, torch.Tensor]) -> torch.Tensor:
        v = Net.search_value(out).float()
        if self.moves_left_alpha:
            m = (out["moves_left"].float() / self.moves_left_scale).clamp(max=1.0)
            v = v * (1 - self.moves_left_alpha * m)
        return v

    def set_model(self, model: Net) -> None:
        if self.graphs:
            with torch.no_grad():
                self.model.load_state_dict(model.state_dict())
        else:
            self.model = model.eval()

    def _forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        autocast = self.device.type == "cuda" and self.model.stem.weight.dtype == torch.float32
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=autocast, cache_enabled=False):
            out = self.model(planes)
        return out["policy"].float(), self.value(out)

    def _graph(self, size: int) -> _Graph:
        if size not in self.graphs:
            planes = torch.zeros((size, FEATURE_BYTES), dtype=torch.uint8, device=self.device)
            side = torch.cuda.Stream(self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(2):  # warm-up: kernel selection, allocations
                    self._forward(planes)
            torch.cuda.current_stream(self.device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            # Other threads (training) keep using the GPU while this thread captures.
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                policy, value = self._forward(planes)
            self.graphs[size] = _Graph(graph, planes, policy, value)
        return self.graphs[size]

    @torch.inference_mode()
    def launch(self, batch: Batch, rows: int) -> None:
        batch.rows = rows
        if rows == 0:
            return
        if self.use_graphs:
            g = self._graph(graph_size(rows))
            g.planes[:rows].copy_(batch.planes[:rows], non_blocking=True)
            g.graph.replay()
            policy, value = g.policy[:rows], g.value[:rows]
        else:
            policy, value = self._forward(batch.planes[:rows].to(self.device, non_blocking=True))
        batch.policy[:rows].copy_(policy, non_blocking=True)
        batch.value[:rows].copy_(value, non_blocking=True)
        if batch.done is not None:
            batch.done.record()
        self.evaluations += rows

    @staticmethod
    def wait(batch: Batch) -> None:
        if batch.done is not None and batch.rows:
            batch.done.synchronize()


def drive(
    pools: Sequence[Pool],
    evaluator: Evaluator,
    should_stop: Callable[[], bool],
    after_step: Callable[[int], None] | None = None,
    before_launch: Callable[[], None] | None = None,
) -> None:
    """Runs the pools round-robin: while the GPU evaluates one pool's leaves, the CPU
    selects leaves in the next. `after_step(i)` runs after pool `i` received results;
    `before_launch()` may swap the evaluator's weights between launches."""
    batches = [Batch(p.max_batch, evaluator.device) for p in pools]
    for pool, batch in zip(pools, batches, strict=True):
        evaluator.launch(batch, pool.collect(batch.planes_np))
    i = 0
    while True:
        pool, batch = pools[i], batches[i]
        evaluator.wait(batch)
        if batch.rows:
            pool.apply(batch.policy_np, batch.value_np, batch.rows)
        if after_step is not None:
            after_step(i)
        if should_stop():
            # Feed back the other pools' launched work, leaving no leaves pending.
            for j in range(1, len(pools)):
                k = (i + j) % len(pools)
                evaluator.wait(batches[k])
                if batches[k].rows:
                    pools[k].apply(batches[k].policy_np, batches[k].value_np, batches[k].rows)
            return
        if before_launch is not None:
            before_launch()
        evaluator.launch(batch, pool.collect(batch.planes_np))
        i = (i + 1) % len(pools)


def evaluate_planes(model: Net, planes: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Synchronous helper: policy logits and search values for a batch of planes."""
    with torch.inference_mode():
        out = model.eval()(torch.from_numpy(planes).to(device))
        return out["policy"].float().cpu().numpy(), Net.search_value(out).float().cpu().numpy()
