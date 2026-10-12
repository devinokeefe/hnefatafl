"""Reinforcement learning from self-play.

One process runs two threads on one GPU: self-play (engine pools evaluated with the
latest fused network) and training (SGD on a growing replay window). A sample-reuse
ratio couples them: training waits for fresh data, and self-play waits when training
falls behind. Everything needed to resume lives in the run directory:

    config.json          settings (edit and restart to change them)
    selfplay/*.npz       sample chunks (games.jsonl holds the game records)
    checkpoints/         model snapshots; latest.pt also holds the optimizer
    metrics.jsonl        one line per logging interval

Usage: python -m tafl.train runs/<name> [--set key=value ...]; create STOP in the
run directory (or press Ctrl+C) to stop cleanly.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import threading
import time
import traceback
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .files import atomic_write
from .inference import Evaluator, drive
from .model import MOVES_LEFT_SCALE, Net, NetConfig
from .native import NUM_ACTIONS, Pool
from .replay import Chunk, ReplayBuffer, TrainingBatch, replay_window


@dataclass
class TrainConfig:
    # Network (fixed for a run).
    blocks: int = 10
    channels: int = 128
    # Self-play.
    pools: int = 2
    games_per_pool: int = 1024
    full_simulations: int = 160
    fast_simulations: int = 40
    full_search_prob: float = 0.25
    considered_actions: int = 16
    # Give defender root moves that threaten an exit fort a first visit (see mcts.rs).
    fort_threats: bool = True
    max_plies: int = 400
    resign_threshold: float = 0.0
    moves_left_alpha: float = 0.15
    moves_left_scale: float = 200.0
    harvest_every: int = 1000
    # Training.
    batch_size: int = 1024
    lr: float = 0.02
    lr_warmup_steps: int = 500
    momentum: float = 0.9
    weight_decay: float = 1e-4
    grad_clip: float = 10.0
    sample_reuse: float = 4.0
    max_training_lag: int = 200_000
    min_samples: int = 100_000
    window_base: int = 250_000
    window_cap: int = 4_000_000
    w_policy: float = 1.0
    w_opponent_policy: float = 0.15
    w_value: float = 1.5
    w_q: float = 0.5
    w_moves_left: float = 0.1
    # Bookkeeping.
    sync_every: int = 40
    log_every: int = 100
    checkpoint_every: int = 2000
    latest_every: int = 250
    max_steps: int = 0
    seed: int = 1

    @staticmethod
    def load(path: Path) -> TrainConfig:
        data = json.loads(path.read_text())
        known = {f.name for f in dataclasses.fields(TrainConfig)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown settings in {path}: {sorted(unknown)}")
        return TrainConfig(**data)

    def save(self, path: Path) -> None:
        text = json.dumps(dataclasses.asdict(self), indent=2) + "\n"
        atomic_write(path, lambda f: f.write(text.encode()))

    def updated(self, assignments: list[str]) -> TrainConfig:
        values = dataclasses.asdict(self)
        for item in assignments:
            key, _, raw = item.partition("=")
            if key not in values:
                raise ValueError(f"unknown setting '{key}'")
            values[key] = type(values[key])(json.loads(raw))
        return TrainConfig(**values)


def _atomic_save(obj: Any, path: Path) -> None:
    atomic_write(path, lambda f: torch.save(obj, f))


def _chunk_number(path: Path) -> int | None:
    """The index in `chunk_000123.npz`, or None for any other file name."""
    _, _, digits = path.stem.partition("_")
    return int(digits) if digits.isdigit() else None


class Trainer:
    def __init__(self, run: Path, config: TrainConfig, device: torch.device) -> None:
        self.run = run
        self.cfg = config
        self.device = device
        (run / "selfplay").mkdir(parents=True, exist_ok=True)
        (run / "checkpoints").mkdir(exist_ok=True)
        torch.manual_seed(config.seed)

        self.model = Net(NetConfig(blocks=config.blocks, channels=config.channels)).to(device)
        decay = [p for p in self.model.parameters() if p.ndim > 1]
        no_decay = [p for p in self.model.parameters() if p.ndim <= 1]
        self.optimizer = torch.optim.SGD(
            [{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=config.lr,
            momentum=config.momentum,
            nesterov=True,
        )
        self.steps = 0
        self.generated = 0
        self.games = 0
        self.chunk_index = 0
        self.restarts = 0
        self.buffer = ReplayBuffer()
        self.stop = threading.Event()
        self.failure: BaseException | None = None
        self.pending_model: Net | None = None
        self.recent_games: deque[dict[str, Any]] = deque(maxlen=2000)
        self.stats_lock = threading.Lock()
        self.evaluations = 0
        self._resume()

    # Persistence.

    def _load_state(self) -> dict[str, Any] | None:
        """The resumable state: latest.pt or, if it cannot be read (a crash while it was
        written), the newest readable snapshot with a fresh optimizer."""
        folder = self.run / "checkpoints"
        latest = folder / "latest.pt"
        if latest.exists():
            try:
                return torch.load(latest, map_location=self.device, weights_only=False)
            except Exception as e:
                self.log(f"cannot read {latest.name}: {e}")
        for path in sorted(folder.glob("step_*.pt"), reverse=True):
            try:
                data = torch.load(path, map_location=self.device, weights_only=False)
            except Exception as e:
                self.log(f"cannot read {path.name}: {e}")
                continue
            self.log(f"resuming from {path.name} with a fresh optimizer")
            state = {"net": data["config"], "model": data["model"], "steps": data["steps"], "games": data["games"]}
            if "chunk_index" in data:
                state.update(generated=data["generated"], chunk_index=data["chunk_index"], restarts=data["restarts"])
            else:  # older snapshots: count every chunk again
                state.update(generated=0, chunk_index=0, restarts=0)
            return state
        return None

    def _resume(self) -> None:
        data = self._load_state()
        if data is not None:
            if data["net"] != NetConfig(blocks=self.cfg.blocks, channels=self.cfg.channels).to_dict():
                raise ValueError("the network shape in config.json differs from the checkpoint")
            self.model.load_state_dict(data["model"])
            if "optimizer" in data:
                self.optimizer.load_state_dict(data["optimizer"])
            self.steps, self.generated, self.games = data["steps"], data["generated"], data["games"]
            self.chunk_index, self.restarts = data["chunk_index"], data["restarts"] + 1
        files: list[tuple[int, Path]] = []
        for path in (self.run / "selfplay").glob("chunk_*.npz"):
            index = _chunk_number(path)
            if index is not None:
                files.append((index, path))
        files.sort()
        for index, path in files:
            if index > self.chunk_index:
                try:
                    with np.load(path) as data:
                        self.generated += len(data["offsets"]) - 1
                except Exception as e:
                    self.log(f"skipping unreadable {path.name}: {e}")
        keep = self._window()
        loaded: list[Chunk] = []
        total = 0
        for _, path in reversed(files):
            if total >= keep:
                break
            try:
                chunk = Chunk.load(path)
            except Exception as e:
                self.log(f"skipping unreadable {path.name}: {e}")
                continue
            loaded.append(chunk)
            total += len(chunk)
        for chunk in reversed(loaded):
            self.buffer.add(chunk, keep)
        if files:
            self.chunk_index = max(self.chunk_index, files[-1][0])
        games = self.run / "selfplay" / "games.jsonl"
        if games.exists() and games.stat().st_size:
            with open(games, "rb+") as f:  # end a line torn by a crash before appending
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.write(b"\n")
        self.log(f"start: step {self.steps}, {self.generated:,} samples generated, {len(self.buffer):,} loaded")

    def _window(self) -> int:
        return replay_window(self.generated, self.cfg.window_base, cap=self.cfg.window_cap)

    def checkpoint(self, snapshot: bool = True) -> None:
        """Saves the resumable state (latest.pt) and, with `snapshot`, a model file."""
        net = self.model.config.to_dict()
        with self.stats_lock:
            counters = {
                "steps": self.steps,
                "generated": self.generated,
                "games": self.games,
                "chunk_index": self.chunk_index,
                "restarts": self.restarts,
            }
        if snapshot:
            _atomic_save(
                {"config": net, "model": self.model.state_dict(), **counters},
                self.run / "checkpoints" / f"step_{self.steps:07d}.pt",
            )
        _atomic_save(
            {"net": net, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), **counters},
            self.run / "checkpoints" / "latest.pt",
        )

    def log(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with open(self.run / "log.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # Self-play.

    def _pool_config(self, index: int) -> dict[str, Any]:
        c = self.cfg
        return {
            "games": c.games_per_pool,
            "full_simulations": c.full_simulations,
            "fast_simulations": c.fast_simulations,
            "full_search_prob": c.full_search_prob,
            "max_plies": c.max_plies,
            "resign_threshold": c.resign_threshold,
            "seed": int(np.random.SeedSequence([c.seed, self.restarts, self.steps, index]).generate_state(1)[0]),
            "search": {
                "algorithm": "gumbel",
                "considered_actions": c.considered_actions,
                "fort_threats": c.fort_threats,
            },
        }

    def _harvest(self, pool: Pool) -> None:
        records, actions, probs, games = pool.take()
        if not games:
            return
        with open(self.run / "selfplay" / "games.jsonl", "a", encoding="utf-8") as f:
            for g in games:
                f.write(json.dumps(g) + "\n")
        with self.stats_lock:
            self.games += len(games)
            self.recent_games.extend(games)
        if len(records) == 0:
            return
        chunk = Chunk.build(records, actions, probs)
        with self.stats_lock:
            self.chunk_index += 1
            self.generated += len(chunk)
            index = self.chunk_index
        chunk.save(self.run / "selfplay" / f"chunk_{index:06d}.npz")
        self.buffer.add(chunk, self._window())

    def _training_behind(self) -> bool:
        if len(self.buffer) < self.cfg.min_samples:
            return False
        lag = self.cfg.sample_reuse * self.generated - self.steps * self.cfg.batch_size
        return lag > self.cfg.sample_reuse * self.cfg.max_training_lag

    def selfplay(self, model: Net) -> None:
        try:
            pools = [Pool(**self._pool_config(i)) for i in range(self.cfg.pools)]
            evaluator = Evaluator(model, self.device, self.cfg.moves_left_alpha, self.cfg.moves_left_scale)
            counter = [0]

            def after_step(i: int) -> None:
                counter[0] += 1
                if counter[0] % self.cfg.harvest_every < len(pools):
                    self._harvest(pools[i])
                while self._training_behind() and not self.stop.is_set():
                    time.sleep(0.05)

            def before_launch() -> None:
                model, self.pending_model = self.pending_model, None
                if model is not None:
                    evaluator.set_model(model)
                self.evaluations = evaluator.evaluations

            drive(pools, evaluator, self.stop.is_set, after_step, before_launch)
            for pool in pools:
                self._harvest(pool)
                pool.close()
        except BaseException as e:
            self.failure = e
            traceback.print_exc()
            self.stop.set()

    # Training.

    def _lr(self) -> float:
        warm = self.cfg.lr_warmup_steps
        return self.cfg.lr * min(1.0, (self.steps + 1) / warm) if warm else self.cfg.lr

    def train_step(self, batch: TrainingBatch) -> dict[str, float] | None:
        """One SGD step; None (and no update) when the loss or gradient is not finite."""
        dev = self.device
        b = len(batch.value)
        planes = torch.from_numpy(batch.planes).to(dev, non_blocking=True)
        target = torch.zeros(b, NUM_ACTIONS, device=dev)
        target[torch.from_numpy(batch.policy_rows).to(dev), torch.from_numpy(batch.policy_actions).to(dev)] = (
            torch.from_numpy(batch.policy_probs).to(dev)
        )
        target /= target.sum(1, keepdim=True).clamp_min(1e-6)
        next_action = torch.from_numpy(batch.next_action).to(dev)
        value = torch.from_numpy(batch.value).to(dev)
        known = torch.from_numpy(batch.result_known).to(dev)
        root_q = torch.from_numpy(batch.root_q).to(dev)
        plies_left = torch.from_numpy(batch.plies_left).to(dev)

        self.model.train()
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = self.model(planes)
        logp = F.log_softmax(out["policy"].float(), 1)
        policy = -(target * logp).sum(1).mean()
        entropy = -(target * target.clamp_min(1e-12).log()).sum(1).mean()
        has_next = next_action >= 0
        opponent = (
            F.cross_entropy(out["opponent_policy"].float()[has_next], next_action[has_next])
            if bool(has_next.any())
            else torch.zeros((), device=dev)
        )
        n_known = known.sum().clamp_min(1.0)
        value_loss = (
            F.binary_cross_entropy_with_logits(out["value_logit"].float(), (value + 1) / 2, reduction="none") * known
        ).sum() / n_known
        q = F.mse_loss(out["q"].float(), root_q)
        moves_left = (
            F.smooth_l1_loss(
                out["moves_left"].float() / MOVES_LEFT_SCALE, plies_left / MOVES_LEFT_SCALE, reduction="none"
            )
            * known
        ).sum() / n_known
        c = self.cfg
        loss = (
            c.w_policy * policy
            + c.w_opponent_policy * opponent
            + c.w_value * value_loss
            + c.w_q * q
            + c.w_moves_left * moves_left
        )
        for group in self.optimizer.param_groups:
            group["lr"] = self._lr()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
        stats = torch.stack([loss, policy, policy - entropy, opponent, value_loss, q, moves_left, grad_norm]).tolist()
        names = ["loss", "policy", "policy_kl", "opponent", "value", "q", "moves_left", "grad_norm"]
        result = dict(zip(names, stats, strict=True))
        if not all(math.isfinite(v) for v in stats):
            # A corrupt sample (or a numeric blow-up): leave the weights untouched.
            self.optimizer.zero_grad(set_to_none=True)
            return None
        self.optimizer.step()
        self.steps += 1
        return result

    def training(self) -> None:
        try:
            rng = np.random.default_rng(self.cfg.seed * 7919 + self.restarts)
            totals: Counter[str] = Counter()
            count = skipped = 0
            mark = (time.time(), self.steps, self.generated, self.evaluations, self.games)
            while not self.stop.is_set():
                if self.cfg.max_steps and self.steps >= self.cfg.max_steps:
                    self.stop.set()
                    break
                allowed = self.cfg.sample_reuse * self.generated
                if len(self.buffer) < self.cfg.min_samples or self.steps * self.cfg.batch_size >= allowed:
                    time.sleep(0.05)
                    continue
                batch = self.buffer.sample(self.cfg.batch_size, self._window(), rng)
                stats = self.train_step(batch)
                if stats is None:
                    skipped += 1
                    self.log(f"skipped a non-finite update at step {self.steps} ({skipped} in a row)")
                    if skipped >= 20:
                        raise FloatingPointError("20 non-finite updates in a row")
                    continue
                skipped = 0
                totals.update(stats)
                count += 1
                if self.steps % self.cfg.sync_every == 0:
                    self.pending_model = self.model.fused()
                if self.steps % self.cfg.log_every == 0:
                    mark = self._report({k: v / count for k, v in totals.items()}, mark)
                    totals.clear()
                    count = 0
                if self.steps % self.cfg.checkpoint_every == 0:
                    self.checkpoint()
                elif self.steps % self.cfg.latest_every == 0:
                    self.checkpoint(snapshot=False)
        except BaseException as e:
            self.failure = e
            traceback.print_exc()
            self.stop.set()

    def _report(self, losses: dict[str, float], mark: tuple[float, ...]) -> tuple[float, ...]:
        now = time.time()
        t0, s0, g0, e0, n0 = mark
        dt = max(now - t0, 1e-9)
        with self.stats_lock:
            games = list(self.recent_games)
        plies = [g["plies"] for g in games]
        winners = Counter(g["winner"] for g in games)
        reasons = Counter(g["reason"] for g in games)
        decided = winners["attackers"] + winners["defenders"]
        row = {
            "time": now,
            "step": self.steps,
            "generated": self.generated,
            "games": self.games,
            "buffer": len(self.buffer),
            "window": self._window(),
            "lr": self._lr(),
            "steps_per_s": (self.steps - s0) / dt,
            "samples_per_s": (self.generated - g0) / dt,
            "evals_per_s": (self.evaluations - e0) / dt,
            "games_per_s": (self.games - n0) / dt,
            "mean_plies": float(np.mean(plies)) if plies else 0.0,
            "attacker_wins": winners["attackers"] / decided if decided else 0.0,
            "undecided": winners[None] / len(games) if games else 0.0,
            "reasons": dict(reasons),
            **losses,
        }
        with open(self.run / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        self.log(
            f"step {self.steps} | samples {self.generated:,} games {self.games:,} | "
            f"{row['evals_per_s']:,.0f} evals/s {row['samples_per_s']:,.0f} samples/s | "
            f"plies {row['mean_plies']:.0f} att {row['attacker_wins']:.2f} | "
            f"loss {losses['loss']:.3f} pol {losses['policy']:.3f} kl {losses['policy_kl']:.3f} "
            f"val {losses['value']:.3f} q {losses['q']:.3f}"
        )
        return (now, self.steps, self.generated, self.evaluations, self.games)

    def run_forever(self) -> None:
        threads = [
            threading.Thread(target=self.selfplay, args=(self.model.fused(),), name="selfplay"),
            threading.Thread(target=self.training, name="training"),
        ]
        for t in threads:
            t.start()
        stop_file = self.run / "STOP"
        try:
            while not self.stop.is_set():
                if stop_file.exists():
                    self.log("STOP file found")
                    stop_file.unlink()
                    self.stop.set()
                time.sleep(1.0)
        except KeyboardInterrupt:
            self.log("interrupted")
            self.stop.set()
        for t in threads:
            t.join()
        self.checkpoint()
        self.log(f"stopped at step {self.steps}" + (f" after an error: {self.failure!r}" if self.failure else ""))
        if self.failure:
            raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="run directory (created if missing)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="change settings (saved)")
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    path = args.run / "config.json"
    config = TrainConfig.load(path) if path.exists() else TrainConfig()
    config = config.updated(args.set)
    if args.set or not path.exists():
        config.save(path)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ.setdefault("RAYON_NUM_THREADS", "16")
    Trainer(args.run, config, device).run_forever()


if __name__ == "__main__":
    main()
