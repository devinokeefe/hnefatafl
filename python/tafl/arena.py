"""Matches between engines: our network search against TaflZero (UCI) or another network.

Games start from random short openings, each played twice with the sides swapped, so
the opening and the side advantage cancel within a pair. Every move is checked by
our referee (whose rules match TaflZero's exactly). Statistics treat the pair as the
independent unit.

    python -m tafl.arena --model runs/r1/checkpoints/step_0100000.pt --opponent taflzero \
        --pairs 100 --nodes 400 --out runs/arena/r1-100k-vs-tz.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import functools
import json
import math
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch

from .inference import Batch, Evaluator
from .model import load_model
from .native import ROOT, Pool, game_info


@dataclass(frozen=True)
class Limit:
    """Search budget per move: nodes, or wall-clock milliseconds."""

    nodes: int = 0
    movetime_ms: int = 0

    def __str__(self) -> str:
        return f"{self.nodes} nodes" if self.nodes else f"{self.movetime_ms} ms"


@dataclass
class MoveResult:
    move: str
    nodes: int = 0
    time_ms: float = 0.0
    value: float | None = None


class Player(Protocol):
    name: str

    def request(self, game: int, moves: Sequence[str]) -> None: ...

    def step(self) -> list[tuple[int, MoveResult]]: ...

    def close(self) -> None: ...


# --- UCI engines (TaflZero) ---


class UciEngine:
    def __init__(self, command: Sequence[str], env: dict[str, str], cwd: Path, creationflags: int = 0) -> None:
        self.proc = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=cwd,
            env=env,
            creationflags=creationflags,
        )
        self.lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.send("uci")
        self._wait_for("uciok", 120)
        self.send("isready")
        self._wait_for("readyok", 120)

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.put(line.rstrip("\r\n"))
        self.lines.put(None)

    def send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, prefix: str, timeout: float) -> tuple[str, list[str]]:
        deadline = time.monotonic() + timeout
        seen: list[str] = []
        while True:
            try:
                line = self.lines.get(timeout=max(deadline - time.monotonic(), 0.01))
            except queue.Empty:
                raise TimeoutError(f"engine did not answer '{prefix}' in {timeout}s") from None
            if line is None:
                raise RuntimeError(f"engine exited (code {self.proc.poll()}); last output: {seen[-5:]}")
            if line.startswith(prefix):
                return line, seen
            seen.append(line)

    def go(self, moves: Sequence[str], limit: Limit) -> MoveResult:
        self.send("position startpos" + (" moves " + " ".join(moves) if moves else ""))
        start = time.perf_counter()
        self.send(f"go nodes {limit.nodes}" if limit.nodes else f"go movetime {limit.movetime_ms}")
        timeout = 600 + (limit.movetime_ms / 1000) * 5
        line, seen = self._wait_for("bestmove", timeout)
        elapsed = (time.perf_counter() - start) * 1000
        info = next((s for s in reversed(seen) if s.startswith("info ")), "")
        fields = info.split()
        nodes = int(fields[fields.index("nodes") + 1]) if "nodes" in fields else 0
        return MoveResult(line.split()[1], nodes, elapsed)

    def close(self) -> None:
        try:
            self.send("quit")
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


@dataclass
class TaflZeroSetup:
    """Where the TaflZero executable, network and ONNX Runtime live."""

    exe: str
    net: str
    ort_dll: str
    extra_path: list[str] = field(default_factory=list)
    ort_threads: int = 1

    @staticmethod
    def load(path: Path) -> TaflZeroSetup:
        data = json.loads(path.read_text())
        data.pop("probe", None)  # used by the rules difftest only
        return TaflZeroSetup(**data)

    def environment(self) -> dict[str, str]:
        env = dict(os.environ)
        env["ORT_DYLIB_PATH"] = self.ort_dll
        env["TAFLZERO_ORT_THREADS"] = str(self.ort_threads)
        env["PATH"] = os.pathsep.join([str(Path(self.ort_dll).parent), *self.extra_path, env.get("PATH", "")])
        return env


class UciPlayer:
    """One engine process per concurrent game, queried from worker threads."""

    def __init__(
        self, name: str, setup: TaflZeroSetup, limit: Limit, workers: int, workdir: Path, creationflags: int = 0
    ) -> None:
        self.name = name
        self.limit = limit
        workdir.mkdir(parents=True, exist_ok=True)
        env = setup.environment()
        command = [setup.exe, "--net", setup.net]
        with concurrent.futures.ThreadPoolExecutor(workers) as ex:
            self.idle = list(ex.map(lambda _: UciEngine(command, env, workdir, creationflags), range(workers)))
        self.executor = concurrent.futures.ThreadPoolExecutor(workers)
        self.running: dict[concurrent.futures.Future[MoveResult], tuple[int, UciEngine]] = {}

    def speed(self, moves: Sequence[str], movetime_ms: int) -> float:
        """Nodes per second of an idle engine searching the position after `moves`."""
        result = self.idle[-1].go(list(moves), Limit(movetime_ms=movetime_ms))
        return 1000 * result.nodes / max(result.time_ms, 1.0)

    def request(self, game: int, moves: Sequence[str]) -> None:
        engine = self.idle.pop()
        self.running[self.executor.submit(engine.go, list(moves), self.limit)] = (game, engine)

    def step(self) -> list[tuple[int, MoveResult]]:
        if not self.running:
            return []
        done, _ = concurrent.futures.wait(self.running, timeout=0.002, return_when=concurrent.futures.FIRST_COMPLETED)
        out = []
        for future in done:
            game, engine = self.running.pop(future)
            self.idle.append(engine)
            out.append((game, future.result()))
        return out

    def close(self) -> None:
        self.executor.shutdown(wait=True)
        for engine in self.idle:
            engine.close()


# --- Our engine ---


class NetPlayer:
    """Searches for many games at once in one analysis pool, batching their leaves."""

    def __init__(
        self,
        name: str,
        model_path: str,
        limit: Limit,
        slots: int,
        device: torch.device,
        search: dict[str, Any],
        leaves_per_search: int = 1,
        moves_left_alpha: float = 0.0,
        reuse_tree: bool = False,
    ) -> None:
        self.name = name
        self.limit = limit
        model, _ = load_model(model_path, device)
        net = model.fused() if device.type == "cuda" else model.eval()
        self.evaluator = Evaluator(net, device, moves_left_alpha)
        self.pool = Pool(
            games=slots, autoplay=False, leaves_per_search=leaves_per_search, search=search, reuse_tree=reuse_tree
        )
        self.batch = Batch(self.pool.max_batch, device)
        self.free = list(range(slots))
        self.searched: list[list[str]] = [[] for _ in range(slots)]  # last game searched per slot
        self.active: dict[int, tuple[int, float]] = {}  # slot -> (game, start time)

    def request(self, game: int, moves: Sequence[str]) -> None:
        # Prefer the slot that searched this game last, so its tree can be reused.
        earlier = [s for s in self.free if list(moves[: len(self.searched[s])]) == self.searched[s]]
        slot = max(earlier, key=lambda s: len(self.searched[s])) if earlier else self.free[-1]
        self.free.remove(slot)
        self.searched[slot] = list(moves)
        self.pool.set_game(slot, "startpos", " ".join(moves))
        budget = self.limit.nodes if self.limit.nodes else 10_000_000
        self.pool.start_search(slot, budget)
        self.active[slot] = (game, time.perf_counter())

    def step(self) -> list[tuple[int, MoveResult]]:
        if not self.active:
            return []
        if self.limit.movetime_ms:
            now = time.perf_counter()
            for slot, (_, start) in self.active.items():
                if (now - start) * 1000 >= self.limit.movetime_ms:
                    self.pool.stop_search(slot)
        rows = self.pool.collect(self.batch.planes_np)
        self.evaluator.launch(self.batch, rows)
        self.evaluator.wait(self.batch)
        if rows:
            self.pool.apply(self.batch.policy_np, self.batch.value_np, rows)
        out = []
        for slot in [s for s in self.active if not self.pool.is_thinking(s)]:
            game, start = self.active.pop(slot)
            self.free.append(slot)
            result = self.pool.state(slot)["result"]
            elapsed = (time.perf_counter() - start) * 1000
            out.append((game, MoveResult(result["best"], result["new_simulations"], elapsed, result["value"])))
        return out

    def close(self) -> None:
        self.pool.close()


# --- Openings and statistics ---


def random_openings(count: int, plies: int, seed: int) -> list[list[str]]:
    """Distinct random openings of `plies` moves that do not end the game."""
    rng = random.Random(seed)
    seen: set[tuple[str, ...]] = set()
    out: list[list[str]] = []
    attempts = 0
    while len(out) < count:
        attempts += 1
        if attempts > count * 1000:
            raise RuntimeError("could not find enough distinct openings")
        moves: list[str] = []
        info = game_info()
        for _ in range(plies):
            moves.append(rng.choice(info["legal"]))
            info = game_info("startpos", " ".join(moves))
            if info["outcome"] is not None:
                break
        if info["outcome"] is None and tuple(moves) not in seen:
            seen.add(tuple(moves))
            out.append(moves)
    return out


def elo(score: float) -> float:
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / score - 1)


def summarize(games: list[dict[str, Any]], player: str) -> dict[str, Any]:
    """Score of `player`, with a confidence interval from per-pair scores."""
    pairs: dict[int, list[float]] = {}
    w = lost = unknown = 0
    by_side: dict[str, list[float]] = {"attackers": [], "defenders": []}
    for g in games:
        side = "attackers" if g["attackers"] == player else "defenders"
        if g["winner"] is None:
            s = 0.5
            unknown += 1
        elif g["winner"] == side:
            s = 1.0
            w += 1
        else:
            s = 0.0
            lost += 1
        by_side[side].append(s)
        pairs.setdefault(g["pair"], []).append(s)
    complete = [sum(p) / 2 for p in pairs.values() if len(p) == 2]
    n = len(complete)
    out: dict[str, Any] = {
        "games": len(games),
        "wins": w,
        "losses": lost,
        "unknown": unknown,
        "as_attackers": sum(by_side["attackers"]) / max(len(by_side["attackers"]), 1),
        "as_defenders": sum(by_side["defenders"]) / max(len(by_side["defenders"]), 1),
        "pairs": n,
    }
    if n:
        mean = sum(complete) / n
        var = sum((x - mean) ** 2 for x in complete) / max(n - 1, 1)
        se = math.sqrt(var / n)
        out.update(
            score=mean,
            score_ci95=(mean - 1.96 * se, mean + 1.96 * se),
            elo=elo(mean),
            elo_ci95=(elo(mean - 1.96 * se), elo(mean + 1.96 * se)),
            z=(mean - 0.5) / se if se > 0 else None,
            pairs_won=sum(1 for x in complete if x > 0.5),
            pairs_lost=sum(1 for x in complete if x < 0.5),
        )
        # Two-sided sign test on decisive pairs.
        k, m = out["pairs_won"], out["pairs_won"] + out["pairs_lost"]
        tail = sum(math.comb(m, i) for i in range(max(k, m - k), m + 1)) / 2**m if m else 1.0
        out["sign_test_p"] = min(1.0, 2 * tail)
    return out


# --- A busy machine ---

BUSY_EXIT = 75  # exit status when the machine stayed too busy; the match itself is intact
PROBE = ("h1h5", "e7e9", "a4c4", "e5e2")  # an ordinary early position for speed checks
INTERRUPTED = "Interrupted"  # how a game ends when it is stopped because the machine got busy


@dataclass(frozen=True)
class SpeedRule:
    """Keeps a busy machine out of the result. A game counts only if `player` averaged
    `nodes_per_s` over its moves and the machine was not seen busy while it was played:
    no other process used the GPU (the record's "busy" flag) and the player's speed
    never dipped, that is, no `window` searches in a row averaged below `dip` times
    `nodes_per_s`. Other games stay in the record but are played again; one that was
    merely slow on average, at most `attempts` times. The rule never looks at who won."""

    player: str
    nodes_per_s: float
    attempts: int = 5
    window: int = 5
    dip: float = 0.75

    def speed(self, game: dict[str, Any]) -> float:
        return 1000 * game[f"{self.player}_mean_nodes"] / max(game[f"{self.player}_mean_ms"], 1e-9)

    def dipped(self, searches: Sequence[Sequence[float]]) -> bool:
        """Whether `window` consecutive searches, given as (nodes, milliseconds), fell
        below the dip speed. Moves made at once (a forced reply, a proven result) are
        left out."""
        real = [(nodes, ms) for nodes, ms in searches if ms >= 1000]
        runs = (real[i : i + self.window] for i in range(len(real) - self.window + 1))
        return any(
            1000 * sum(n for n, _ in run) / sum(ms for _, ms in run) < self.dip * self.nodes_per_s for run in runs
        )

    def disturbed(self, game: dict[str, Any]) -> bool:
        return bool(game.get("busy")) or self.dipped(game.get(f"{self.player}_searches", ()))

    def counts(self, game: dict[str, Any]) -> bool:
        return not self.disturbed(game) and self.speed(game) >= self.nodes_per_s


def other_gpu_load(ours: Collection[int]) -> float:
    """Percent of the GPU's 3D and compute engines in use by processes other than
    `ours`, from the Windows performance counters; 0 where those cannot be read."""
    command = ["typeperf", r"\GPU Engine(*)\Utilization Percentage", "-sc", "1"]
    try:
        text = subprocess.run(command, capture_output=True, text=True, timeout=300).stdout
        names, values = [row[1:] for row in csv.reader(text.splitlines()) if len(row) > 1][:2]
        total = 0.0
        for name, value in zip(names, values, strict=False):
            engine = re.search(r"pid_(\d+)_.*engtype_(3D|Cuda|Compute)", name)
            if engine and int(engine.group(1)) not in ours and value.strip():
                total += float(value)
        return total
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0.0


class GpuWatch:
    """Reads, on a background thread, how much of the GPU other processes are using.
    The GPU counts as taken when two readings in a row are above `limit` percent, so a
    moment of desktop activity does not stop a game."""

    def __init__(
        self,
        ours: Collection[int],
        limit: float = 15.0,
        pause_s: float = 20.0,
        read: Callable[[Collection[int]], float] = other_gpu_load,
    ) -> None:
        self.limit = limit
        self.readings: list[tuple[float, float]] = []  # (time the reading began, percent), newest last
        self.fresh = threading.Condition()
        threading.Thread(target=self._run, args=(set(ours), pause_s, read), daemon=True).start()

    def _run(self, ours: set[int], pause_s: float, read: Callable[[Collection[int]], float]) -> None:
        while True:
            began = time.perf_counter()  # finer than the monotonic clock's 16 ms on Windows
            load = read(ours)
            with self.fresh:
                self.readings = [*self.readings[-1:], (began, load)]
                self.fresh.notify_all()
            time.sleep(pause_s)

    def taken(self) -> bool:
        """Whether the last two readings were both above the limit."""
        last = self.readings
        return len(last) == 2 and min(load for _, load in last) >= self.limit

    def taken_now(self, timeout_s: float = 600.0) -> bool:
        """The same, after waiting for a reading that began after this call."""
        asked = time.perf_counter()
        with self.fresh:
            self.fresh.wait_for(lambda: bool(self.readings) and self.readings[-1][0] >= asked, timeout_s)
        return self.taken()


class Busy(Exception):
    """Other work on the machine keeps an engine below the speed a fair game needs."""


def wait_for_quiet(engine: UciPlayer, rule: SpeedRule, patience_s: float, watch: GpuWatch | None = None) -> None:
    """Returns once no other process holds the GPU and an idle engine reaches the rule's
    speed on a test position, checking once a minute; raises Busy if the machine is
    still not quiet after `patience_s`."""
    deadline = time.monotonic() + patience_s
    while True:
        if watch and watch.taken_now():
            print("machine busy: another process is using the GPU", flush=True)
        else:
            speed = engine.speed(PROBE, 5000)
            if speed >= rule.nodes_per_s:
                print(f"machine quiet: {engine.name} reaches {speed:.0f} nodes/s", flush=True)
                return
            print(f"machine busy: {engine.name} reaches {speed:.0f} of {rule.nodes_per_s:.0f} nodes/s", flush=True)
        if time.monotonic() >= deadline:
            raise Busy
        time.sleep(60)


def above_normal_priority() -> int:
    """Moves this process above the normal priority class, and returns the flag that
    starts a child process there, so that background jobs on a shared machine do not
    take the engines' processor time. Does nothing outside Windows."""
    if os.name != "nt":
        return 0
    import ctypes

    flag = subprocess.ABOVE_NORMAL_PRIORITY_CLASS
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), flag):
        raise ctypes.WinError(ctypes.get_last_error())
    return flag


def realtime_gpu_priority(pids: Collection[int]) -> None:
    """Puts the processes in the realtime class of the Windows GPU scheduler, where
    their GPU work runs ahead of every other process's. An engine then searches as fast
    as on an idle machine while other jobs use the GPU. Those jobs pay for it: while an
    engine is searching they are left with a few percent of the GPU."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    set_class = ctypes.WinDLL("gdi32").D3DKMTSetProcessSchedulingPriorityClass
    set_class.restype = ctypes.c_int32
    set_class.argtypes = [wintypes.HANDLE, ctypes.c_int]
    for pid in pids:
        process = kernel32.OpenProcess(0x0200, False, pid)  # PROCESS_SET_INFORMATION
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            status = set_class(process, 5)  # D3DKMT_SCHEDULINGPRIORITYCLASS_REALTIME
        finally:
            kernel32.CloseHandle(process)
        if status:
            raise OSError(f"GPU priority of process {pid} not set (NTSTATUS {status & 0xFFFFFFFF:#010x})")


# --- The match loop ---


def play_match(
    a: Player,
    b: Player,
    openings: list[list[str]],
    concurrency: int,
    max_plies: int,
    log_path: Path | None,
    done: list[dict[str, Any]] | None = None,
    rule: SpeedRule | None = None,
    before_game: Callable[[], None] | None = None,
    watch: GpuWatch | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Plays each opening twice (each player attacking once), `concurrency` games at a time.
    Games in `done` (records of an earlier, interrupted run) are not played again.
    `before_game` runs whenever a game is about to start with no other game in progress.
    With a `rule`, games are stopped when `watch` finds the GPU taken by another process.
    Returns the games that count and the number that `rule` set aside."""
    schedule = []
    for i, opening in enumerate(openings):
        schedule.append({"pair": i, "opening": opening, "attackers": a.name, "defenders": b.name})
        schedule.append({"pair": i, "opening": opening, "attackers": b.name, "defenders": a.name})
    finished: list[dict[str, Any]] = []
    aside = 0  # recorded games that do not count
    slow: dict[tuple[int, str], int] = {}  # of those, the ones that were just slow, per scheduled game

    def set_aside(g: dict[str, Any]) -> bool:
        """Notes a game that does not count; True if it is to be played again."""
        nonlocal aside
        aside += 1
        if rule is not None and rule.disturbed(g):
            return True
        key = (g["pair"], g["attackers"])
        slow[key] = slow.get(key, 0) + 1
        return rule is not None and slow[key] < rule.attempts

    for g in done or []:
        key = {
            "pair": g["pair"],
            "opening": g["opening"].split(),
            "attackers": g["attackers"],
            "defenders": g["defenders"],
        }
        if key not in schedule:
            continue
        if rule is None or rule.counts(g):
            schedule.remove(key)
            finished.append(g)
        elif not set_aside(g):
            schedule.remove(key)
    total = len(schedule) + len(finished)
    players = {a.name: a, b.name: b}
    pending = list(reversed(schedule))
    live: dict[int, dict[str, Any]] = {}
    next_id = 0
    start = time.time()

    def ask(gid: int) -> None:
        g = live[gid]
        side = "attackers" if len(g["moves"]) % 2 == 0 else "defenders"
        players[g[side]].request(gid, g["moves"])

    while pending or live:
        while pending and len(live) < concurrency:
            if before_game and not live:
                before_game()
            g = pending.pop()
            g.update(moves=list(g["opening"]), stats={a.name: [], b.name: []})
            live[next_id] = g
            ask(next_id)
            next_id += 1
        if rule and watch and watch.taken():
            for g in live.values():
                g["busy"] = True  # each stops at its next move
        for player in (a, b):
            for gid, result in player.step():
                g = live[gid]
                g["stats"][player.name].append((result.nodes, result.time_ms))
                if rule and player.name == rule.player and rule.dipped(g["stats"][player.name][-rule.window :]):
                    g["busy"] = True  # something slowed the engine down; the game stops here
                try:
                    info = game_info("startpos", " ".join([*g["moves"], result.move]))
                except Exception as e:
                    info = {"outcome": {"winner": None, "reason": f"illegal move {result.move} by {player.name}: {e}"}}
                    g["error"] = str(e)
                else:
                    g["moves"].append(result.move)
                if info["outcome"] is None and len(g["moves"]) >= max_plies:
                    info["outcome"] = {"winner": None, "reason": "PlyCap"}
                if info["outcome"] is None and g.get("busy"):
                    info["outcome"] = {"winner": None, "reason": INTERRUPTED}
                if info["outcome"] is None:
                    ask(gid)
                    continue
                if rule and watch and not g.get("busy") and watch.taken_now():
                    g["busy"] = True  # the GPU was taken during the last moves
                g["winner"] = info["outcome"]["winner"]
                g["reason"] = info["outcome"]["reason"]
                g["plies"] = len(g["moves"])
                for name, stats in g.pop("stats").items():
                    g[f"{name}_mean_nodes"] = sum(s[0] for s in stats) / max(len(stats), 1)
                    g[f"{name}_mean_ms"] = sum(s[1] for s in stats) / max(len(stats), 1)
                    g[f"{name}_searches"] = [[nodes, round(ms)] for nodes, ms in stats]
                again = {k: g[k] for k in ("pair", "opening", "attackers", "defenders")}
                g["moves"] = " ".join(g["moves"])
                g["opening"] = " ".join(g["opening"])
                del live[gid]
                if log_path:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(g) + "\n")
                note = ""
                if rule is None or rule.counts(g):
                    finished.append(g)
                else:
                    speed = f"{rule.player} at {rule.speed(g):.0f} nodes/s"
                    note = " | not counted: " + ("the machine was busy" if rule.disturbed(g) else speed)
                    if set_aside(g):
                        pending.append(again)
                s = summarize(finished, a.name)
                winner = g[g["winner"]] if g["winner"] else "unknown"
                print(
                    f"[{time.time() - start:7.0f}s] game {len(finished)}/{total}: {g['attackers']} (att) vs "
                    f"{g['defenders']} (def) -> {winner} by {g['reason']} in {g['plies']} plies | "
                    f"{a.name} +{s['wins']} -{s['losses']} ={s['unknown']}{note}",
                    flush=True,
                )
    return finished, aside


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="checkpoint of the engine under test")
    parser.add_argument("--opponent", default="taflzero", help="'taflzero' or a checkpoint path")
    parser.add_argument("--taflzero-config", type=Path, default=ROOT / "taflzero.local.json")
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--opening-plies", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=0, help="nodes per move for both sides")
    parser.add_argument("--movetime", type=int, default=0, help="milliseconds per move for both sides")
    parser.add_argument("--opponent-nodes", type=int, default=0)
    parser.add_argument("--opponent-movetime", type=int, default=0)
    parser.add_argument("--algorithm", default="puct", choices=["puct", "gumbel"])
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--leaves", type=int, default=1, help="leaves per search (virtual loss)")
    parser.add_argument("--moves-left-alpha", type=float, default=0.0, help="game-length shaping of values")
    parser.add_argument(
        "--no-reuse-tree",
        action="store_true",
        help="search every move from scratch (by default the tree is kept between moves, as TaflZero does; "
        "node budgets count new work for both engines)",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=600)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="keep the games already recorded in --out")
    parser.add_argument(
        "--min-opponent-speed",
        type=float,
        default=0.0,
        help="nodes per second TaflZero must average for a game to count; slower games are recorded and "
        "played again, and each game waits for a test search to reach this speed (for a shared machine)",
    )
    parser.add_argument("--attempts", type=int, default=3, help="times a game is played before it is given up")
    parser.add_argument("--patience", type=float, default=10.0, help="minutes to wait for a quiet machine")
    parser.add_argument("--above-normal", action="store_true", help="run both engines above normal priority")
    parser.add_argument(
        "--gpu-priority",
        action="store_true",
        help="run both engines ahead of all other work on the GPU (Windows): games are then played while "
        "other processes use it, and those processes get almost none of it during a search",
    )
    args = parser.parse_args()
    if bool(args.nodes) == bool(args.movetime):
        parser.error("give exactly one of --nodes and --movetime")
    if args.min_opponent_speed and args.opponent != "taflzero":
        parser.error("--min-opponent-speed needs the TaflZero opponent")

    creationflags = above_normal_priority() if args.above_normal else 0
    if args.gpu_priority:
        realtime_gpu_priority([os.getpid()])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ours = Limit(args.nodes, args.movetime)
    theirs = ours
    if args.opponent_nodes or args.opponent_movetime:
        theirs = Limit(args.opponent_nodes, args.opponent_movetime)
    search = {"algorithm": args.algorithm, "gumbel_noise": False, "c_puct": args.c_puct}
    reuse = not args.no_reuse_tree
    common = (args.concurrency, device, search, args.leaves, args.moves_left_alpha, reuse)
    a = NetPlayer("ours", args.model, ours, *common)
    b: Player
    rule: SpeedRule | None = None
    before_game: Callable[[], None] | None = None
    watch: GpuWatch | None = None
    if args.opponent == "taflzero":
        setup = TaflZeroSetup.load(args.taflzero_config)
        workdir = ROOT / "runs" / "arena" / "taflzero-cwd"
        b = taflzero = UciPlayer("taflzero", setup, theirs, args.concurrency, workdir, creationflags)
        engines = [engine.proc.pid for engine in taflzero.idle]
        if args.gpu_priority:
            realtime_gpu_priority(engines)
        if args.min_opponent_speed:
            rule = SpeedRule(taflzero.name, args.min_opponent_speed, args.attempts)
            if not args.gpu_priority:  # with it, other work on the GPU waits for the engines
                watch = GpuWatch({os.getpid(), *engines})
            before_game = functools.partial(wait_for_quiet, taflzero, rule, args.patience * 60, watch)
    else:
        b = NetPlayer("opponent", args.opponent, theirs, *common)
    openings = random_openings(args.pairs, args.opening_plies, args.seed)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "model": args.model,
        "opponent": args.opponent,
        "ours": str(ours),
        "theirs": str(theirs),
        "search": search,
        "leaves": args.leaves,
        "reuse_tree": reuse,
        "moves_left_alpha": args.moves_left_alpha,
        "pairs": args.pairs,
        "opening_plies": args.opening_plies,
        "seed": args.seed,
        "min_opponent_speed": args.min_opponent_speed,
        "above_normal": args.above_normal,
        "gpu_priority": args.gpu_priority,
    }
    print(json.dumps(header), flush=True)
    done = []
    if args.resume and args.out and args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line torn by a crash
            if "reason" in record:
                done.append(record)
        aside = sum(not rule.counts(g) for g in done) if rule else 0
        print(
            f"resuming with {len(done)} recorded games" + (f", {aside} of them not counted" * bool(aside)), flush=True
        )
    try:
        games, set_aside = play_match(
            a, b, openings, args.concurrency, args.max_plies, args.out, done, rule, before_game, watch
        )
    except Busy:
        print("the machine stayed busy; run again with --resume to continue", flush=True)
        sys.exit(BUSY_EXIT)
    finally:
        a.close()
        b.close()
    summary = {**header, **summarize(games, "ours"), "not_counted": set_aside}
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
