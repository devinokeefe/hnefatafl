"""Replay buffer of self-play samples, with symmetry-augmented batch sampling."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .files import atomic_write
from .native import NO_ACTION, NUM_SYMMETRIES, SAMPLE_DTYPE, encode_states, symmetry_actions


@dataclass
class Chunk:
    """Samples of some finished games; sample `i` owns policy entries `offsets[i]:offsets[i+1]`."""

    records: np.ndarray
    offsets: np.ndarray
    actions: np.ndarray
    probs: np.ndarray

    @staticmethod
    def build(records: np.ndarray, actions: np.ndarray, probs: np.ndarray) -> Chunk:
        offsets = np.zeros(len(records) + 1, np.int64)
        np.cumsum(records["policy_len"], out=offsets[1:])
        if offsets[-1] != len(actions):
            raise ValueError("policy entries do not match the records")
        return Chunk(records, offsets, actions.astype(np.uint16), probs.astype(np.float16))

    def __len__(self) -> int:
        return len(self.records)

    def merged(self, other: Chunk) -> Chunk:
        return Chunk(
            np.concatenate([self.records, other.records]),
            np.concatenate([self.offsets, other.offsets[1:] + self.offsets[-1]]),
            np.concatenate([self.actions, other.actions]),
            np.concatenate([self.probs, other.probs]),
        )

    def save(self, path: Path) -> None:
        arrays = {"records": self.records, "offsets": self.offsets, "actions": self.actions, "probs": self.probs}
        atomic_write(path, lambda f: np.savez(f, **arrays))

    @staticmethod
    def load(path: Path) -> Chunk:
        with np.load(path) as data:
            return Chunk(data["records"].view(SAMPLE_DTYPE), data["offsets"], data["actions"], data["probs"])


@dataclass
class TrainingBatch:
    planes: np.ndarray  # uint8 [b, FEATURE_BYTES]
    policy_rows: np.ndarray  # int64 [e]: sparse policy target entries
    policy_actions: np.ndarray  # int64 [e]
    policy_probs: np.ndarray  # float32 [e]
    next_action: np.ndarray  # int64 [b], -1 when absent
    value: np.ndarray  # float32 [b], game result for the side to move
    result_known: np.ndarray  # float32 [b]
    root_q: np.ndarray  # float32 [b]
    plies_left: np.ndarray  # float32 [b]


def replay_window(total: int, base: int, alpha: float = 0.75, beta: float = 0.4, cap: int | None = None) -> int:
    """KataGo-style growing window: all data up to `base`, then sublinear growth."""
    size = total if total <= base else int(base * (1 + beta * ((total / base) ** alpha - 1) / alpha))
    return min(size, cap) if cap else size


class ReplayBuffer:
    """Chunks of samples, oldest first. Small chunks are merged into blocks of about
    `block_size` samples so that sampling touches few arrays."""

    def __init__(self, block_size: int = 65_536) -> None:
        self.block_size = block_size
        self.chunks: deque[Chunk] = deque()
        self.size = 0
        self.lock = threading.Lock()
        self.symmetry = symmetry_actions().astype(np.int64)

    def __len__(self) -> int:
        return self.size

    def add(self, chunk: Chunk, keep: int) -> None:
        """Adds a chunk and drops the oldest ones not needed to keep `keep` samples."""
        with self.lock:
            if self.chunks and len(self.chunks[-1]) < self.block_size:
                chunk = self.chunks.pop().merged(chunk)
            self.chunks.append(chunk)
            self.size = sum(len(c) for c in self.chunks)
            while self.chunks and self.size - len(self.chunks[0]) >= keep:
                self.size -= len(self.chunks.popleft())

    def sample(self, batch: int, window: int, rng: np.random.Generator) -> TrainingBatch:
        """Uniform samples from the newest `window` samples, each under a random symmetry."""
        with self.lock:
            chunks = list(self.chunks)
        sizes = np.array([len(c) for c in chunks], np.int64)
        starts = np.concatenate([[0], np.cumsum(sizes)])
        total = int(starts[-1])
        window = min(window, total)
        index = np.sort(rng.integers(total - window, total, batch))
        owner = np.searchsorted(starts, index, side="right") - 1

        records = np.empty(batch, SAMPLE_DTYPE)
        rows, actions, probs = [], [], []
        for c in np.unique(owner):
            sel = np.nonzero(owner == c)[0]
            local = index[sel] - starts[c]
            chunk = chunks[c]
            records[sel] = chunk.records[local]
            begin = chunk.offsets[local]
            counts = chunk.offsets[local + 1] - begin
            n = int(counts.sum())
            entry = np.repeat(begin - np.concatenate([[0], np.cumsum(counts)[:-1]]), counts) + np.arange(n)
            rows.append(np.repeat(sel, counts))
            actions.append(chunk.actions[entry])
            probs.append(chunk.probs[entry])

        sym = rng.integers(0, NUM_SYMMETRIES, batch).astype(np.uint8)
        planes = encode_states(np.ascontiguousarray(records["state"]), sym)
        row = np.concatenate(rows)
        mapped = self.symmetry[sym[row], np.concatenate(actions).astype(np.int64)]
        next_action = records["next_action"].astype(np.int64)
        present = next_action != NO_ACTION
        next_action[present] = self.symmetry[sym[present], next_action[present]]
        next_action[~present] = -1
        return TrainingBatch(
            planes=planes,
            policy_rows=row.astype(np.int64),
            policy_actions=mapped,
            policy_probs=np.concatenate(probs).astype(np.float32),
            next_action=next_action,
            value=records["value"].astype(np.float32),
            result_known=records["result_known"].astype(np.float32),
            root_q=records["root_q"].astype(np.float32),
            plies_left=records["plies_left"].astype(np.float32),
        )
