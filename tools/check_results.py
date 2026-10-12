"""Checks the numbers that README.md, docs/RESULTS.md and CHANGELOG.md state against the records.

    python tools/check_results.py              print what the records give, match by match
    python tools/check_results.py --check      check every stated number; exit with status 1 on a problem
    python tools/check_results.py --json PATH  write every game, whether it counts and why not, and each
                                               match's counted results, for figures (JSON_NOTE below)

The records are the matches in matches/ (tools/copy_records.py), the training records in training/
and models/manifest.json, which identifies the two checkpoints the results use (tools/training_records.py).
From them this script
recomputes which games of each match count, the results, scores, Elo intervals, pairs and sign tests,
the results by side and by how the games ended, game lengths, both engines' speeds, the games that
did not count and why, the speeds that the logs quote, and the training steps, samples, games and
times that the text quotes. It also checks the records against each other: each match's summary as
the arena wrote it against the recomputed one, the game records against their SHA-256, the manifest
against the training metrics, and a checkpoint against the manifest, the configuration and the code where
a copy has been put in models/ (the checkpoints are not distributed).

`stated` lists the fragments of the documents that state such numbers, written from the
computed values; each must occur in its document word for word (line breaks and runs of spaces
aside). --check prints "agrees with the records: <file>" for a document that has all of them, and
"MISSING in <file>: <fragment>" for each that is not there; "DIFFERS: ..." where two records
disagree. NOT_CHECKED lists what the documents state that the records cannot show, with the reason;
those fragments must still be in the text, so that the list stays complete. A match in PENDING (one
still being played) is computed and its fragments are looked for, but one that is missing does not
fail the check.

Much of this script repeats the wording of the documents: whoever changes a sentence there that
states a number must change the fragment here.

It needs only the standard library. Which games count is decided as tafl.arena decides it when a
match is resumed from its records (SpeedRule, play_match and summarize). Those few lines are repeated
here because tafl.arena imports PyTorch, and must stay the same: python/tests/test_check_results.py
compares the two where PyTorch is installed.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import pickle
import re
import statistics
import sys
import zipfile
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MATCHES = ROOT / "matches"
TRAINING = ROOT / "training"
MODELS = ROOT / "models"
DIFFTEST = ROOT / "docs" / "difftest.txt"
README = "README.md"
RESULTS = "docs/RESULTS.md"
CHANGELOG = "CHANGELOG.md"
RUN = "r1"

# Matches still being played. Their numbers are computed and printed and their fragments looked for,
# but a missing fragment does not fail the check. Take a match out of this set once its row is final.
PENDING: set[str] = set()

# --attempts of tafl.arena, its default. The headers do not record it; with these records any value
# from 2 up gives the same games (count_games checks that), since no game was slow twice.
ATTEMPTS = 3
OURS = "ours"
GAME_LINE = re.compile(r"^\[\s*(\d+)s\] game ")
QUIET_LINE = re.compile(r"^machine quiet: taflzero reaches (\d+) nodes/s$")
WORDS = dict(
    enumerate(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        ]
    )
)


def word(n: int) -> str:
    """A number in words, as the text writes small numbers (in figures past twenty)."""
    return WORDS.get(n, f"{n:,}")


# --- The arena's rules (tafl.arena), repeated without PyTorch ---


@dataclass(frozen=True)
class SpeedRule:
    """tafl.arena.SpeedRule: a game counts only if `player` averaged `nodes_per_s`, no other process
    used the GPU while it was played (the "busy" flag) and the speed never dipped."""

    player: str
    nodes_per_s: float
    attempts: int = 5
    window: int = 5
    dip: float = 0.75

    def speed(self, game: dict[str, Any]) -> float:
        return 1000 * game[f"{self.player}_mean_nodes"] / max(game[f"{self.player}_mean_ms"], 1e-9)

    def dipped(self, searches: Sequence[Sequence[float]]) -> bool:
        real = [(nodes, ms) for nodes, ms in searches if ms >= 1000]
        runs = (real[i : i + self.window] for i in range(len(real) - self.window + 1))
        return any(
            1000 * sum(n for n, _ in run) / sum(ms for _, ms in run) < self.dip * self.nodes_per_s for run in runs
        )

    def disturbed(self, game: dict[str, Any]) -> bool:
        return bool(game.get("busy")) or self.dipped(game.get(f"{self.player}_searches", ()))

    def counts(self, game: dict[str, Any]) -> bool:
        return not self.disturbed(game) and self.speed(game) >= self.nodes_per_s


def elo(score: float) -> float:
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / score - 1)


def summarize(games: list[dict[str, Any]], player: str) -> dict[str, Any]:
    """tafl.arena.summarize: the score of `player`, with a confidence interval from per-pair scores."""
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
        k, m = out["pairs_won"], out["pairs_won"] + out["pairs_lost"]
        tail = sum(math.comb(m, i) for i in range(max(k, m - k), m + 1)) / 2**m if m else 1.0
        out["sign_test_p"] = min(1.0, 2 * tail)
    return out


@dataclass
class Counting:
    """Which records of a match count, as tafl.arena.play_match decides when it resumes from them:
    the first record of each scheduled game (pair and attacking side) that the rule lets count; one
    that does not is set aside and the game played again, unless it was merely slow (not disturbed)
    `attempts` times. Records of a game that had already counted or been given up are not looked at."""

    counted: list[dict[str, Any]]
    aside: list[tuple[dict[str, Any], str]]  # with why: "slow", "busy" or "dipped"
    unused: list[dict[str, Any]]


def count_games(games: list[dict[str, Any]], header: dict[str, Any], rule: SpeedRule | None) -> Counting:
    opponent = "taflzero" if header["opponent"] == "taflzero" else "opponent"
    schedule = {(pair, side) for pair in range(header["pairs"]) for side in (OURS, opponent)}
    counted, aside, unused = [], [], []
    slow: Counter[tuple[int, str]] = Counter()
    for g in games:
        key = (g["pair"], g["attackers"])
        if key not in schedule:
            unused.append(g)
        elif rule is None or rule.counts(g):
            schedule.remove(key)
            counted.append(g)
        else:
            if g.get("busy"):
                aside.append((g, "busy"))
            elif rule.disturbed(g):
                aside.append((g, "dipped"))
            else:
                aside.append((g, "slow"))
                slow[key] += 1
                if slow[key] >= rule.attempts:
                    schedule.remove(key)
    return Counting(counted, aside, unused)


# --- The records ---


def read_games(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().decode("utf-8").splitlines() if line.strip()]


@dataclass
class Run:
    """One run of the arena for a match (a match continued with --resume has several)."""

    log: str
    created: datetime
    last_written: datetime
    header: dict[str, Any]
    lines: list[str]  # its log
    games: list[dict[str, Any]]  # the records it added

    def quiet(self) -> list[tuple[int, datetime]]:
        """The speeds of the test searches before games ("machine quiet"), with about when each was
        made: the time of the game line before it, counted from when the log was created."""
        out, seconds = [], 0
        for line in self.lines:
            if game := GAME_LINE.match(line):
                seconds = int(game[1])
            elif quiet := QUIET_LINE.match(line):
                out.append((int(quiet[1]), self.created + timedelta(seconds=seconds)))
        return out


@dataclass
class Match:
    name: str
    runs: list[Run]
    games: list[dict[str, Any]]
    header: dict[str, Any]  # the last run's, whose rule decides which games count
    rule: SpeedRule | None
    counting: Counting
    summary: dict[str, Any]
    record: dict[str, Any]  # matches/<name>.json

    @property
    def counted(self) -> list[dict[str, Any]]:
        return self.counting.counted

    @property
    def opponent(self) -> str:
        return "taflzero" if self.header["opponent"] == "taflzero" else "opponent"


def rule_of(header: dict[str, Any], attempts: int = ATTEMPTS) -> SpeedRule | None:
    speed = header.get("min_opponent_speed") or 0.0
    return SpeedRule("taflzero", speed, attempts) if speed else None


def load_match(name: str) -> Match:
    record = json.loads((MATCHES / f"{name}.json").read_text(encoding="utf-8"))
    games = read_games(MATCHES / record["records"]["file"])
    runs, start = [], 0
    for i, entry in enumerate(record["runs"]):
        lines = (MATCHES / entry["log"]).read_text(encoding="utf-8").splitlines()
        played = sum(1 for line in lines if GAME_LINE.match(line))
        # A game is recorded just before the arena logs it: the last run may hold one more.
        end = len(games) if i == len(record["runs"]) - 1 else start + played
        runs.append(
            Run(
                entry["log"],
                datetime.fromisoformat(entry["created"]),
                datetime.fromisoformat(entry["last_written"]),
                entry["header"],
                lines,
                games[start:end],
            )
        )
        start = end
    header = runs[-1].header
    rule = rule_of(header)
    counting = count_games(games, header, rule)
    return Match(name, runs, games, header, rule, counting, summarize(counting.counted, OURS), record)


@dataclass
class Training:
    metrics: list[dict[str, float]]
    events: list[tuple[datetime, str]]
    selfplay: list[dict[str, int | float | str]]
    config: dict[str, Any]

    def row(self, step: int) -> dict[str, float]:
        return next(r for r in self.metrics if r["step"] == step)

    def at_or_before(self, step: int) -> dict[str, float]:
        return [r for r in self.metrics if r["step"] <= step][-1]

    def event(self, pattern: str) -> tuple[datetime, re.Match[str]]:
        """The first event whose message matches `pattern`."""
        for when, message in self.events:
            if found := re.fullmatch(pattern, message):
                return when, found
        raise LookupError(f"no event {pattern!r} in training/{RUN}-events.txt")

    def started(self, step: int) -> tuple[datetime, int]:
        """When training (re)started at `step`, and the samples generated by then."""
        when, found = self.event(rf"start: step {step}, ([\d,]+) samples generated, [\d,]+ loaded")
        return when, int(found[1].replace(",", ""))

    def time_of(self, step: int) -> float:
        """When `step` was reached: its metrics row, or the event of a stop there."""
        rows = [r for r in self.metrics if r["step"] == step]
        if rows:
            return rows[0]["time"]
        return self.event(rf"stopped at step {step}")[0].timestamp()

    def hours(self, step: int) -> float:
        """Hours of training to `step`, counted from the first logged step (training began a few
        seconds before it; the self-play that filled the first window began 12 minutes earlier)."""
        return (self.time_of(step) - self.metrics[0]["time"]) / 3600

    def samples(self, step: int) -> int:
        rows = [r for r in self.metrics if r["step"] == step]
        return int(rows[0]["generated"]) if rows else self.started(step)[1]

    def logged_during(self, runs: Iterable[Run]) -> bool:
        """Whether any training step was logged while one of the runs was in progress."""
        spans = [(run.created.timestamp(), run.last_written.timestamp()) for run in runs]
        return any(a <= row["time"] <= b for row in self.metrics for a, b in spans)


def load_training(zone: timezone) -> Training:
    with open(TRAINING / f"{RUN}-metrics.csv", encoding="utf-8", newline="") as f:
        metrics = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(f)]
    events = []
    for line in (TRAINING / f"{RUN}-events.txt").read_text(encoding="utf-8").splitlines():
        stamp, message = line[:19], line[20:]
        events.append((datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=zone), message))
    with open(TRAINING / f"{RUN}-selfplay.csv", encoding="utf-8", newline="") as f:
        selfplay = [
            {key: (value if key == "approx_step" else float(value)) for key, value in row.items()}
            for row in csv.DictReader(f)
        ]
    config = json.loads((TRAINING / f"{RUN}-config.json").read_text(encoding="utf-8"))
    return Training(metrics, events, selfplay, config)


# --- Checkpoints, read without PyTorch ---


class _Stub:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        pass


@dataclass(frozen=True)
class Shape:
    size: tuple[int, ...]


def _rebuild_tensor(storage: Any, offset: int, size: Sequence[int], *args: Any) -> Shape:
    return Shape(tuple(size))


class _Checkpoint(pickle.Unpickler):
    """Reads the pickle of a checkpoint that torch.save wrote, without PyTorch: a tensor becomes its
    shape, any other object a stub, so only plain values and shapes are read, and no code is run."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor
        return _Stub

    def persistent_load(self, pid: Any) -> None:
        return None


def read_checkpoint(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
        return _Checkpoint(io.BytesIO(archive.read(name))).load()


# --- Writing numbers as the documents do ---


def num(x: float, step: float = 1) -> str:
    """x rounded half up to a multiple of `step`, with thousands separators."""
    value = math.floor(x / step + 0.5) * step
    decimals = max(0, -int(Decimal(str(step)).normalize().as_tuple().exponent))
    return f"{value:,.{decimals}f}"


def two(x: float) -> str:
    """A score: two decimals, rounded half up."""
    return str(Decimal(repr(x)).quantize(Decimal("0.01"), ROUND_HALF_UP))


def elo_text(x: float) -> str:
    rounded = math.floor(x + 0.5)
    return "0" if rounded == 0 else f"{rounded:+d}"


def elo_with_interval(s: dict[str, Any]) -> str:
    low, high = s["elo_ci95"]
    return f"Elo {elo_text(s['elo'])} [{elo_text(low)}, {elo_text(high)}]"


def p_text(p: float) -> str:
    """A p-value: one significant digit below 0.001, three decimals below 0.1, else two."""
    if p < 0.001:
        mantissa, exponent = f"{p:.0e}".split("e")
        return f"{mantissa}e{int(exponent)}"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def millions(n: float) -> str:
    return f"{n / 1e6:.1f}M"


def span(values: Iterable[float], step: float = 1, sep: str = "-") -> str:
    values = list(values)
    return f"{num(min(values), step)}{sep}{num(max(values), step)}"


def game_numbers(numbers: Sequence[int]) -> str:
    """Game numbers as the text writes them: "1-4", "9-10", "5-8 and 11"."""
    runs: list[list[int]] = []
    for n in sorted(numbers):
        if runs and n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    parts = [f"{r[0]}-{r[-1]}" if len(r) > 1 else str(r[0]) for r in runs]
    return " and ".join([", ".join(parts[:-1]), parts[-1]] if len(parts) > 1 else parts)


def dates(runs: Sequence[Run]) -> str:
    """The date of a match, or its first and last dates if it ran over several."""
    first, last = runs[0].created.date(), runs[-1].last_written.date()
    return first.isoformat() if first == last else f"{first.isoformat()} to {last:%m-%d}"


def when(moment: datetime) -> str:
    return f"{moment:%Y-%m-%d %H:%M}"


def consistent(condition: bool, fragment: str, why: str = "") -> str:
    """A fragment whose words the records bear out only as a whole: it is looked for as it stands if
    they do, and reported as missing, with what the records show instead, if they do not."""
    return fragment if condition else f"{fragment} [the records do not bear this out{': ' + why if why else ''}]"


# --- What the records show about games ---


def outcome(g: dict[str, Any]) -> str:
    """The result of a game for our engine: win, loss or unfinished."""
    if g["winner"] is None:
        return "unfinished"
    side = "attackers" if g["attackers"] == OURS else "defenders"
    return "win" if g["winner"] == side else "loss"


def as_side(games: Iterable[dict[str, Any]], side: str) -> list[dict[str, Any]]:
    """The games in which our engine played `side`."""
    return [g for g in games if (g["attackers"] == OURS) == (side == "attackers")]


def tally(games: Iterable[dict[str, Any]], result: str) -> Counter[str]:
    """How the games with that result for us ended."""
    return Counter(g["reason"] for g in games if outcome(g) == result)


NAMES = {  # reason: (one, several)
    "KingCaptured": ("king capture", "king captures"),
    "Encircled": ("encirclement", "encirclements"),
    "NoMoves": ("no moves", "no moves"),
    "Repetition": ("repetition", "repetitions"),
    "ExitFort": ("exit fort", "exit forts"),
    "KingEscaped": ("escape", "escapes"),
}


def listing(counts: Counter[str], order: Sequence[str], names: dict[str, tuple[str, str]] | None = None) -> str:
    """'3 king captures, 2 encirclements' for the reasons in `order`; a reason not in `order` is
    added at the end, so that a fragment with it no longer matches the text."""
    names = {**NAMES, **(names or {})}
    parts = []
    for reason in [*order, *sorted(set(counts) - set(order), key=str)]:
        n = counts.get(reason, 0)
        if n:
            one, several = names.get(reason, (str(reason), str(reason)))
            parts.append(f"{n} {one if n == 1 else several}")
    return ", ".join(parts)


def wl(s: dict[str, Any]) -> str:
    """'12-1, 3 unfinished'."""
    return f"{s['wins']}-{s['losses']}" + (f", {s['unknown']} unfinished" if s["unknown"] else "")


def speed(g: dict[str, Any], player: str) -> float:
    return 1000 * g[f"{player}_mean_nodes"] / max(g[f"{player}_mean_ms"], 1e-9)


def median_nodes(games: Iterable[dict[str, Any]], player: str) -> float:
    """The median over games of the nodes per move that a player averaged."""
    return statistics.median(g[f"{player}_mean_nodes"] for g in games)


def complete_pairs(games: Iterable[dict[str, Any]]) -> int:
    return sum(1 for n in Counter(g["pair"] for g in games).values() if n == 2)


def same_setup(a: Match, b: Match) -> bool:
    """Whether two matches had the same settings, apart from the network, and the same openings."""
    opening = {g["pair"]: g["opening"] for g in b.games}
    return {k: v for k, v in a.header.items() if k != "model"} == {
        k: v for k, v in b.header.items() if k != "model"
    } and all(g["opening"] == opening.get(g["pair"]) for g in a.games)


def step_of(model: str) -> int:
    return int(re.search(r"step_(\d+)\.pt", model)[1])  # type: ignore[index]


def budget(header: dict[str, Any]) -> str:
    ours, theirs = header["ours"], header["theirs"]
    if ours != theirs:
        return f"{ours} against {theirs}"
    if ours.endswith(" ms"):
        return f"{int(ours.split()[0]) // 1000} s per move"
    return ours


def dip_stretch(g: dict[str, Any], rule: SpeedRule) -> tuple[float, float]:
    """The longest run of consecutive full searches of the rule's player in a game that were each
    slower than the dip rule's threshold: their speed together, and how long the run lasted in
    seconds, from the end of the search before it to the start of the search after it (so with our
    searches around them)."""
    theirs = g[f"{rule.player}_searches"]
    ours = g[f"{OURS}_searches"]
    slow = [ms >= 1000 and 1000 * nodes / ms < rule.dip * rule.nodes_per_s for nodes, ms in theirs]
    runs: list[tuple[int, int]] = []
    for i, is_slow in enumerate(slow):
        if is_slow and runs and runs[-1][1] == i - 1:
            runs[-1] = (runs[-1][0], i)
        elif is_slow:
            runs.append((i, i))
    if not runs:
        return 0.0, 0.0
    a, b = max(runs, key=lambda run: run[1] - run[0])
    # Our search i comes before theirs i when we attack (the attackers move first), after it when we defend.
    before = 1 if g["attackers"] == OURS else 0
    ours_between = range(max(a + before - 1, 0), min(b + before, len(ours) - 1) + 1)
    ms = sum(theirs[i][1] for i in range(a, b + 1))
    nodes = sum(theirs[i][0] for i in range(a, b + 1))
    return 1000 * nodes / ms, (ms + sum(ours[i][1] for i in ours_between)) / 1000


def first_search_slowdown(match: Match) -> dict[str, float]:
    """How much slower the first search of a replay game was in the busier half of those games than in
    the quieter half, by the test speed before each, for each engine (mean of the halves)."""
    pairs = []
    for run in match.runs[1:]:
        readings, n = [], 0
        for line in run.lines:
            if quiet := QUIET_LINE.match(line):
                readings.append(int(quiet[1]))
            elif GAME_LINE.match(line):
                if readings and n < len(run.games):
                    pairs.append((readings[-1], run.games[n]))
                readings, n = [], n + 1
    pairs.sort(key=lambda p: p[0])
    half = len(pairs) // 2
    busier, quieter = pairs[:half], pairs[len(pairs) - half :]
    out = {}
    for player in ("taflzero", OURS):

        def first(g: dict[str, Any], player: str = player) -> float:
            nodes, ms = g[f"{player}_searches"][0]
            return nodes / ms

        out[player] = 1 - statistics.mean(first(g) for _, g in busier) / statistics.mean(first(g) for _, g in quieter)
    return out


# --- The statements of the documents ---


@dataclass
class Fragment:
    file: str
    text: str
    match: str | None = None  # the match it is about, for PENDING


@dataclass
class Unchecked:
    file: str
    text: str
    why: str


NOT_CHECKED = [
    Unchecked(README, "running its network g0213", "which engine and network the opponent was is not in the records"),
    Unchecked(
        README,
        "(the time limit of the Fetlar world championship)",
        "a rule of the championship, not a record",
    ),
    Unchecked(
        README,
        "gives 116, 6,788 and 806,344 for one, two and three plies.",
        "computed by `tafl perft` and asserted by the test perft_from_the_start in crates/tafl-cli, not a record",
    ),
    Unchecked(README, "(the same size as TaflZero's)", "TaflZero's network is not in the records"),
    Unchecked(
        README,
        "TaflZero's 11 input planes have no move history and, for repetition, mark only how often the position "
        "itself has occurred.",
        "TaflZero's code is not in the records",
    ),
    Unchecked(
        README,
        "the one playtafl.org's page was loading when it was downloaded",
        "where the network came from is not in the records",
    ),
    Unchecked(
        README,
        "(meant as 80% of its speed on a quiet machine)",
        "the speed on the idle machine was never recorded",
    ),
    Unchecked(
        README,
        "upstream commit `9b5a19a` with 41 local commits",
        "TaflZero's build is not in the records; it is described from its own repository on the machine that "
        "played the matches",
    ),
    Unchecked(
        README,
        "The repetition and fort tactics were added after matches against TaflZero",
        "when each tactic was added is in the project's history, not in the records",
    ),
    Unchecked(
        README,
        "over 60 GB",
        "measured in the run's folder, which is not copied; training/README.md gives the sizes",
    ),
    Unchecked(
        README,
        "an Intel i9-14900K (32 threads), an NVIDIA RTX 4090 and 64 GB of memory",
        "the machine is not in the records",
    ),
    Unchecked(
        README,
        "about ten hours each at 15 seconds a move",
        "a rough figure: the first run of the step-36,000 match took 9.7 hours for 16 games, by its log",
    ),
    Unchecked(
        RESULTS,
        "(the build described in the README, with the g0213 network)",
        "which engine and network the opponent was",
    ),
    Unchecked(
        RESULTS,
        "ours did not before 2026-10-03 22:30)",
        "when the change was made is not recorded; the headers agree (no tree reuse in the matches that "
        "began before then, reuse in those after), which the check of the s6000 row tests",
    ),
    Unchecked(
        RESULTS,
        "The limit is 80% of the speed TaflZero reaches on the idle machine.",
        "the speed on the idle machine was never recorded",
    ),
    Unchecked(RESULTS, "(idle about 1,500)", "not recorded; the logs' test speeds were taken on a busy machine"),
    Unchecked(RESULTS, "where idle is about 1,700", "not recorded; the logs' test speeds were taken on a busy machine"),
    Unchecked(
        RESULTS,
        "Node counts in the notes and records up to 2026-10-10 are not comparable between the",
        "what the engines counted is a fact about the code, not in the records",
    ),
    Unchecked(RESULTS, "From 2026-10-11 our count is the simulations of the search itself.", "as above"),
    Unchecked(RESULTS, "before CUDA-graph inference", "a fact about the code"),
    Unchecked(RESULTS, "which led to the repetition tactic", "a fact about the code"),
    Unchecked(RESULTS, "With the repetition tactic and", "a fact about the code (tree reuse is checked)"),
    Unchecked(
        RESULTS,
        "First match with the exit-fort tactic (from step 22,137).",
        "what changed is not recorded; the training events show a restart at that step, which is checked",
    ),
    Unchecked(RESULTS, "with its original engine library", "which library was loaded is not recorded"),
    Unchecked(
        RESULTS,
        "at a time when other jobs had exhausted the machine's memory",
        "the machine's memory is not recorded (the error output shows TaflZero's process ending)",
    ),
    Unchecked(
        RESULTS,
        "after the fort tactic and fort-threat exploration (step 22,710)",
        "what changed is not recorded; the training events show restarts at steps 22,137 and 22,710, and "
        "the configuration has fort_threats on, which is checked",
    ),
    Unchecked(
        RESULTS,
        "other jobs filled the processor and later the GPU, and slowed both engines",
        "the other jobs are not recorded, only the engines' speeds",
    ),
    Unchecked(RESULTS, "one game at a time", "the arena's --concurrency is not in the headers"),
    Unchecked(
        RESULTS,
        "the first search of a game was 20-25% slower for TaflZero and 13-23% slower for ours",
        "the method is not stated; splitting the replay games by the test speed before each, the records "
        "give the figures that the report prints",
    ),
    Unchecked(
        RESULTS,
        "A restart of the computer at 2026-10-12 00:09 ended the match process during the first game of the "
        "seventh pair; that game left no record.",
        "a game stopped that way leaves no record; the logs show only the gap between the runs",
    ),
    Unchecked(RESULTS, "which is harmless here because both sides are our engine", "a judgement"),
]


def check_records(matches: dict[str, Match], training: Training) -> list[str]:
    """Checks of the records against each other; returns what differs."""
    problems = []
    for m in matches.values():
        data = (MATCHES / m.record["records"]["file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != m.record["records"]["sha256"]:
            problems.append(f"{m.name}.jsonl is not the file whose SHA-256 {m.name}.json gives")
        if len(m.games) != m.record["records"]["games"]:
            problems.append(f"{m.name}.jsonl does not hold the number of games {m.name}.json gives")
        openings = {g["pair"]: g["opening"] for g in m.games}
        if any(openings[g["pair"]] != g["opening"] for g in m.games):
            problems.append(f"{m.name}: the two games of a pair start from different openings")
        alternatives = {
            tuple(id(g) for g in count_games(m.games, m.header, rule_of(m.header, n)).counted) for n in range(2, 6)
        }
        if m.rule and len(alternatives) != 1:
            problems.append(f"{m.name}: the games that count depend on --attempts, which is not recorded")
        for label, arena, games, aside in (
            ("summary", m.record.get("summary"), m.counted, len(m.counting.aside)),
            ("first_run_summary", m.record.get("first_run_summary"), m.runs[0].games, None),
        ):
            if not arena:
                continue
            ours = {**summarize(games, OURS), **({"not_counted": aside} if aside is not None else {})}
            for key, value in arena.items():
                mine = ours.get(key)
                same = (
                    all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(value, mine, strict=True))
                    if isinstance(value, list)
                    else value == mine or (value is not None and mine is not None and math.isclose(value, mine))
                )
                if not same:
                    problems.append(f"{m.name}: the arena's {label} gives {key} = {value}, the records {mine}")
    for entry in manifest():
        row = training.row(entry["step"])
        for key, value in {"samples": row["generated"], "games": row["games"]}.items():
            if entry[key] != value:
                problems.append(f"models/manifest.json gives {key} = {entry[key]} for {entry['file']}, not {value}")
        if abs(datetime.fromisoformat(entry["date"]).timestamp() - row["time"]) > 1:
            problems.append(f"models/manifest.json: the date of {entry['file']} is not that of its step")
        path = MODELS / entry["file"]
        if not path.exists():
            continue  # not distributed: main() lists it as not checked
        data = path.read_bytes()
        checkpoint = read_checkpoint(path)
        for key, value in {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}.items():
            if entry[key] != value:
                problems.append(f"models/manifest.json gives {key} = {entry[key]} for {entry['file']}, not {value}")
        if [checkpoint["steps"], checkpoint["generated"], checkpoint["games"]] != [
            entry["step"],
            row["generated"],
            row["games"],
        ]:
            problems.append(f"{entry['file']}: its own counters are not those of the metrics at its step")
        if network_of(checkpoint) != network(training):
            problems.append(f"{entry['file']}: its network is {network_of(checkpoint)}, not {network(training)}")
    return problems


def manifest() -> list[dict[str, Any]]:
    """The entries of models/manifest.json, one per checkpoint that the results use."""
    return json.loads((MODELS / "manifest.json").read_text(encoding="utf-8"))["models"]


def network(t: Training) -> dict[str, int]:
    """The shape of the network as the configuration and the code give it: blocks and channels from the
    run's configuration, the squeeze-excitation ratio from NetConfig, the input and move planes from the
    engine's encoding. check_records compares it with the checkpoints where they are here."""
    model = (ROOT / "python" / "tafl" / "model.py").read_text(encoding="utf-8")
    encode = (ROOT / "crates" / "tafl-engine" / "src" / "encode.rs").read_text(encoding="utf-8")
    se_ratio = re.search(r"^    se_ratio: int = (\d+)$", model, re.M)
    planes = re.search(r"pub const COUNT: usize = (\d+);", encode)
    actions = re.search(r"pub const NUM_ACTIONS: usize = (\d+) \* NUM_SQUARES;", encode)
    if not (se_ratio and planes and actions):
        raise SystemExit("check_results: the network's shape is no longer where network() looks for it")
    return {
        "blocks": t.config["blocks"],
        "channels": t.config["channels"],
        "se_ratio": int(se_ratio[1]),
        "planes": int(planes[1]),
        "move_planes": int(actions[1]),
    }


def network_of(checkpoint: dict[str, Any]) -> dict[str, int]:
    """The same, as a checkpoint holds it."""
    config, weights = checkpoint["config"], checkpoint["model"]
    return {
        "blocks": config["blocks"],
        "channels": config["channels"],
        "se_ratio": config["se_ratio"],
        "planes": weights["stem.weight"].size[1],
        "move_planes": weights["policy.out.weight"].size[0] // 2,  # the policy and the opponent's reply
    }


def stated(matches: dict[str, Match], t: Training) -> list[Fragment]:
    """The fragments of the documents that state what the records show, as the records give them."""
    m = matches
    out: list[Fragment] = []

    def add(file: str, *texts: str, match: str | None = None) -> None:
        out.extend(Fragment(file, text, match) for text in texts)

    def row_head(x: Match, extra: str = "") -> str:
        """'| 2026-10-04 | r1 step 16,000 (4.1M samples, ~6 h)' style: date and network."""
        return f"| {dates(x.runs)} | {RUN} step {step_of(x.header['model']):,}{extra}"

    def paused(x: Match, runs: Sequence[Run] | None = None) -> str:
        return "training paused" if not t.logged_during(runs or x.runs) else "training running"

    def side_counts(x: Sequence[dict[str, Any]], side: str) -> tuple[Counter[str], Counter[str], Counter[str]]:
        games = as_side(x, side)
        return tally(games, "win"), tally(games, "loss"), tally(games, "unfinished")

    # README: the status.
    one_s, fifteen, last = m["s36000-t1000-paused"], m["s36000-t15000-paused"], m["s468000-t15000-paused"]
    s1, s15, s468 = one_s.summary, fifteen.summary, last.summary
    unfinished_plies = {g["plies"] for g in last.counted if g["winner"] is None}
    days = num(t.hours(468000) / 24, 0.1)
    add(
        README,
        f"At {budget(last.header).replace(' s per move', ' seconds a move')}, the network of step "
        f"{step_of(last.header['model']):,}, after {days} days of training, won {s468['wins']} of {s468['games']} "
        f"games against TaflZero, lost {s468['losses'] or 'none'} and left {s468['unknown']} unfinished at the "
        f"{'/'.join(str(p) for p in sorted(unfinished_plies))}-ply limit: "
        + ("all " if s468["pairs_won"] == s468["pairs"] else f"{s468['pairs_won']} of ")
        + f"{s468['pairs']} opening pairs won (sign test p = {p_text(s468['sign_test_p'])}).",
        consistent(
            same_setup(last, fifteen) and step_of(fifteen.header["model"]) == 36000,
            "With the same openings, the network of step 36,000,",
        )
        + f" after about {num(t.hours(36000))} hours of training, had won {s15['wins']}, lost {s15['losses']} and "
        f"left {s15['unknown']} unfinished: {s15['pairs_won']} pairs won, "
        f"{s15['pairs'] - s15['pairs_won'] - s15['pairs_lost']} tied and "
        f"{'none' if not s15['pairs_lost'] else s15['pairs_lost']} lost (p = {p_text(s15['sign_test_p'])}).",
        consistent(
            step_of(one_s.header["model"]) == step_of(fifteen.header["model"]),
            f"At {budget(one_s.header).replace(' s per move', ' second a move')} that network had won "
            f"{s1['wins']} games, lost {s1['losses']} and left {s1['unknown']} unfinished, with {s1['pairs_won']} of "
            f"{s1['pairs']} pairs won and {s1['pairs_lost']} lost.",
        ),
        consistent(
            s15["games"] == s468["games"], f"{word(s15['games']).capitalize()} games a match are a small sample"
        ),
    )
    add(
        CHANGELOG,
        f"At 15 seconds per move the network of step {step_of(last.header['model']):,} won "
        + ("all " if s468["pairs_won"] == s468["pairs"] else f"{s468['pairs_won']} of ")
        + f"{s468['pairs']} opening pairs against TaflZero, with {s468['wins']} games won, "
        f"{s468['losses'] or 'none'} lost and {s468['unknown']} unfinished.",
    )
    # README: the network and training settings, from the configuration and the code (and the checkpoints,
    # where they are here: check_records).
    net = network(t)
    add(
        README,
        consistent(net["se_ratio"] > 0, f"a {net['blocks']}-block, {net['channels']}-channel residual tower"),
        f"the {net['move_planes'] * 121:,} move actions ({net['move_planes']} move planes over the board)",
        f"Inputs are {net['planes']}\nplanes",
        f"playout-cap randomisation ({num(100 * t.config['full_search_prob'])}% of moves get a full search",
        consistent(
            bool(t.config["fort_threats"]),
            f"up to half of the {t.config['considered_actions']} moves the root considers",
        ),
    )
    # README: the table of the 15-second matches, and the training time to the second one's network.

    def games_at(step: int) -> int:
        return int(next(r["games"] for r in t.metrics if r["step"] == step))

    slow_plays = Counter(
        (name, g["pair"], g["attackers"])
        for name, x in m.items()
        if name.endswith("-t15000-paused")
        for g, why in x.counting.aside
        if why == "slow"
    )
    add(
        README,
        f"| step {step_of(fifteen.header['model']):,} | about {num(t.hours(36000))} hours, "
        f"{games_at(36000):,} games | {s15['wins']}, {s15['losses']}, {s15['unknown']} | {s15['pairs_won']}, "
        f"{s15['pairs'] - s15['pairs_won'] - s15['pairs_lost']}, {s15['pairs_lost']} | "
        f"p = {p_text(s15['sign_test_p'])} |",
        f"| step 468,000 | {days} days, {games_at(468000):,} games | {s468['wins']}, {s468['losses']}, "
        f"{s468['unknown']} | {s468['pairs_won']}, {s468['pairs'] - s468['pairs_won'] - s468['pairs_lost']}, "
        f"{s468['pairs_lost']} | p = {p_text(s468['sign_test_p'])} |",
        f"{days} days to step 468,000",
        f"averaged at least {num(fifteen.rule.nodes_per_s if fifteen.rule else 0)} nodes per second over its moves",
        consistent(
            max(slow_plays.values(), default=0) <= 1,
            f"`--attempts` plays ({ATTEMPTS} by default), and no game was slow twice",
        ),
    )
    # README: the last full run of the rules against TaflZero's, from the record of its output.
    record = DIFFTEST.read_text(encoding="utf-8") if DIFFTEST.exists() else ""
    compared = re.search(r"^compared (\d+) positions in (\d+) games$", record, re.M)
    mismatching = re.search(r"^(\d+) mismatching games$", record, re.M)
    add(
        README,
        f"The last full run: {int(compared[2]):,} games, {int(compared[1]):,} positions, "
        f"{int(mismatching[1])} mismatches."
        if compared and mismatching
        else "The last full run: [docs/difftest.txt has no summary]",
    )

    # RESULTS: the introduction.
    s26, s26p = m["s26000-t1000"], m["s26000-t1000-paused"]
    first_run, replays = fifteen.runs[0], fifteen.runs[1:]
    limits = [run.header.get("min_opponent_speed") for run in replays]
    low_runs = [run for run in replays if run.header.get("min_opponent_speed") == min(limits)]
    high_runs = [run for run in replays if run.header.get("min_opponent_speed") == max(limits)]
    evening = next(run for run in replays if run.created.hour >= 18 and run.created.date() == date(2026, 10, 5))
    evening_quiet = [s for s, _ in evening.quiet()[:2]]
    all_quiet = [s for run in replays for s, _ in run.quiet()]
    busy_runs = [run for match in m.values() for run in match.runs if any(g.get("busy") for g in run.games)]
    first_busy = min(busy_runs, key=lambda run: run.created)
    dips = [g for g, why in fifteen.counting.aside if why == "dipped"]
    dip_speed, dip_seconds = dip_stretch(dips[0], fifteen.rule) if dips else (0.0, 0.0)  # type: ignore[arg-type]
    add(
        RESULTS,
        f"(about {num(median_nodes(s26.games, 'taflzero'), 10)} instead of "
        f"{num(median_nodes(s26p.games, 'taflzero'), 100)} "
        "nodes\nper second at 1 s per move)",
        f"For the 15 s match that was first taken as {num(min(limits))} nodes per",
        consistent(all(run.header.get("above_normal") for run in replays), "replayed games run above normal priority"),
        f"since {when(low_runs[-1].last_written)[:-2]}00 the limit is {num(max(limits))}"
        if high_runs[0].created > low_runs[-1].last_written
        else "since [the records do not bear this out]",
        f"on the evening of {evening.created.date()}, with fewer of them, the same test reached "
        f"{num(min(evening_quiet), 100)} to {num(max(evening_quiet), 10)}.",
        f"From {first_busy.created.date()} the\narena also watches the GPU",
        f"one replayed win had such a stretch (about {num(dip_speed, 100)} nodes per second for "
        f"{word(round(dip_seconds / 60))} minutes)"
        if len(dips) == 1 and outcome(dips[0]) == "win"
        else "one replayed win had such a stretch [the records do not bear this out]",
    )

    add(
        README,
        f"TaflZero searched about {num(median_nodes(s26.games, 'taflzero'), 10)} nodes per move against about "
        f"{num(median_nodes(s26p.games, 'taflzero'), 100)} with training paused",
    )

    # RESULTS: the matches against TaflZero, row by row.
    x = m["s1699-n400"]
    s = x.summary
    tz_defending = as_side(x.games, "attackers")
    escapes = [g["plies"] for g in tz_defending if g["reason"] == "KingEscaped"]
    tz_attacking = as_side(x.games, "defenders")
    no_moves = sum(1 for g in tz_attacking if g["reason"] == "NoMoves" and outcome(g) == "loss")
    tz_defence = Counter(g["reason"] for g in tz_defending)
    add(
        RESULTS,
        row_head(x, " (~"),
        f"~{num((t.time_of(1699) - t.metrics[0]['time']) / 60, 5)} min of training,",
        f"min of training, {num(t.samples(1699) / 1000)}k samples) | {budget(x.header)} | {x.header['pairs']} | "
        f"{wl(s)} | {two(s['score'])} ({s['pairs_lost']} pairs lost, sign test p = {p_text(s['sign_test_p'])}) |",
        consistent(s["losses"] == s["games"], "TaflZero won every game:"),
        consistent(
            tz_defence["KingEscaped"] + tz_defence["ExitFort"] == len(tz_defending)
            and all(outcome(g) == "loss" for g in tz_defending),
            f"as defender by escape in about {num(statistics.median(escapes))} plies "
            f"({tz_defence['KingEscaped']} games) or by an exit fort ({tz_defence['ExitFort']}),",
            f"TaflZero won as defender by {listing(tz_defence, ['KingEscaped', 'ExitFort'])}",
        ),
        consistent(no_moves > len(tz_attacking) / 2, "as attacker mostly by leaving our defenders without a move."),
        f"Our engine used {statistics.mean(g['ours_mean_ms'] for g in x.games) / 1000:.1f} s per move against "
        f"TaflZero's {statistics.mean(g['taflzero_mean_ms'] for g in x.games) / 1000:.1f} s",
        match=x.name,
    )
    previous = x
    x = m["s6000-n400"]
    s = x.summary
    mean_plies = statistics.mean(g["plies"] for g in x.games)
    tz_wins = Counter(g["reason"] for g in as_side(x.games, "defenders"))
    tz_defence6 = Counter(g["reason"] for g in as_side(x.games, "attackers"))
    add(
        RESULTS,
        row_head(x, f" ({millions(t.samples(6000))} samples) | {budget(x.header)} | {x.header['pairs']} | {wl(s)} | ")
        + f"{two(s['score'])} (p = {p_text(s['sign_test_p'])}) |",
        consistent(
            mean_plies > 2 * statistics.mean(g["plies"] for g in previous.games),
            f"Games much longer ({num(mean_plies)} plies on average).",
        ),
        consistent(
            all(
                g["reason"] in ("KingEscaped", "ExitFort") and outcome(g) == "loss"
                for g in as_side(x.games, "attackers")
            ),
            f"As defender TaflZero won every game, {tz_defence6['KingEscaped']} by escape and "
            f"{tz_defence6['ExitFort']} by exit fort;",
            "TaflZero won as defender by " + listing(tz_defence6, ["KingEscaped", "ExitFort"]),
        ),
        f"as attacker it won all {tz_wins['Repetition']} by a third repetition"
        if tz_wins["Repetition"] == len(as_side(x.games, "defenders"))
        else "as attacker it won all [the records do not bear this out]",
        consistent(
            not x.header["reuse_tree"]
            and x.runs[0].created.hour < 23
            and "reuse_tree" not in m["s1699-n400"].header
            and all(y.header.get("reuse_tree") for y in m.values() if y.runs[0].created > x.runs[-1].last_written),
            "No tree reuse on our side.",
        ),
        match=x.name,
    )

    x = m["s10000-n400"]
    s = x.summary
    defending = as_side(x.games, "defenders")
    attacking = as_side(x.games, "attackers")
    add(
        RESULTS,
        row_head(x),
        f"r1 step {step_of(x.header['model']):,} ({millions(t.samples(10000))} samples, ~{num(t.hours(10000))} h of "
        f"training) | {budget(x.header)} | {x.header['pairs']} | {wl(s)} | {two(s['score'])} ({s['pairs_lost']} pairs "
        f"lost, p = {p_text(s['sign_test_p'])}) |",
        consistent(x.header["reuse_tree"], "and tree reuse."),
        consistent(
            all(g["winner"] is None and g["plies"] == 600 for g in defending),
            f"Our defence held all {len(defending)} of TaflZero's attacks to the 600-ply limit;",
        ),
        consistent(
            all(outcome(g) == "loss" and g["reason"] in ("KingEscaped", "ExitFort") for g in attacking),
            f"our attack lost all {len(attacking)} (TaflZero escaped or built an exit fort in "
            f"{span(g['plies'] for g in attacking)} plies).",
        ),
        f"Mean game {num(statistics.mean(g['plies'] for g in x.games))} plies.",
        match=x.name,
    )

    x = m["s16000-n400"]
    s = x.summary
    wins_d, losses_d, held_d = side_counts(x.games, "defenders")
    wins_a, losses_a, _ = side_counts(x.games, "attackers")
    add(
        RESULTS,
        row_head(x, f" ({millions(t.samples(16000))} samples, ~{num(t.hours(16000))} h) | {budget(x.header)} | ")
        + f"{x.header['pairs']} | {wl(s)} | {two(s['score'])}, {elo_with_interval(s)} ({s['pairs_won']} pair won, "
        f"{s['pairs_lost']} lost, p = {p_text(s['sign_test_p'])}) |",
        consistent(
            all(y.summary["wins"] == 0 for y in (m["s1699-n400"], m["s6000-n400"], m["s10000-n400"]))
            and list(wins_a) == ["KingCaptured"]
            and list(wins_d) == ["KingEscaped"],
            f"First wins: {word(wins_a.total())} as attacker (king captured), {word(wins_d.total())} as defender "
            "(escapes).",
        ),
        f"As defender: {held_d.total()} held to 600 plies, {losses_d['Repetition']} lost to repetition."
        if losses_d.total() == losses_d["Repetition"]
        else "As defender: [the records do not bear this out]",
        f"As attacker: {losses_a.total()} of {len(as_side(x.games, 'attackers'))} lost (TaflZero built exit forts in "
        f"{losses_a['ExitFort']}).",
        match=x.name,
    )

    x = m["s18000-t1000"]
    s = x.summary
    wins_d, losses_d, held_d = side_counts(x.games, "defenders")
    wins_a, losses_a, _ = side_counts(x.games, "attackers")
    forts = [g["plies"] for g in as_side(x.games, "attackers") if outcome(g) == "loss" and g["reason"] == "ExitFort"]
    add(
        RESULTS,
        row_head(x, f" ({millions(t.samples(18000))} samples, ~{num(t.hours(18000))} h) | {budget(x.header)} | ")
        + f"{x.header['pairs']} | {wl(s)} | {two(s['score'])}, {elo_with_interval(s)} ({s['pairs_lost']} pairs lost, "
        f"p = {p_text(s['sign_test_p'])}) |",
        f"First timed match, {x.header['leaves']} leaves per batch, "
        f"{paused(x).replace('running', 'running alongside')}.",
        f"Our engine searched about {num(median_nodes(x.games, OURS), 100)} nodes per move to TaflZero's "
        f"{num(median_nodes(x.games, 'taflzero'), 10)}.",
        f"As defender: {held_d.total()} held to 600 plies, {listing(wins_d, ['KingEscaped'])}, "
        f"{losses_d['Repetition']} lost to repetition."
        if losses_d.total() == losses_d["Repetition"]
        else "As defender: [the records do not bear this out]",
        f"As attacker: {wins_a.total()} win (defenders left without a move), {losses_a.total()} losses, "
        f"{len(forts)} of them to exit forts built in {span(forts)} plies."
        if list(wins_a) == ["NoMoves"]
        else "As attacker: [the records do not bear this out]",
        match=x.name,
    )

    x = s26
    s = x.summary
    wins_a, losses_a, _ = side_counts(x.games, "attackers")
    wins_d, losses_d, held_d = side_counts(x.games, "defenders")
    events_22137 = any(msg.startswith("start: step 22137,") for _, msg in t.events)
    add(
        RESULTS,
        row_head(x, f" ({millions(t.samples(26000))} samples,"),
        f"samples, ~{num(t.hours(26000), 0.5)} h) | {budget(x.header)} | {x.header['pairs']} | {wl(s)} | "
        f"{two(s['score'])}, {elo_with_interval(s)} ({s['pairs_won']} pairs won, {s['pairs_lost']} lost, "
        f"p = {p_text(s['sign_test_p'])}) |",
        consistent(events_22137 and step_of(x.header["model"]) > 22137 > 18000, "(from step 22,137)"),
        f"As attacker: {wins_a.total()} wins ("
        + listing(wins_a, ["KingCaptured", "NoMoves"], {"NoMoves": ("by leaving the defenders without a move",) * 2})
        + f"), {losses_a['ExitFort']} losses to exit forts."
        if losses_a.total() == losses_a["ExitFort"]
        else "As attacker: [the records do not bear this out]",
        f"As defender: {listing(wins_d, ['KingEscaped'])}, {held_d.total()} held to 600 plies."
        if not losses_d
        else "As defender: [the records do not bear this out]",
        f"Caveat: {paused(x).replace('training running', 'training ran alongside')} and TaflZero searched only "
        f"~{num(median_nodes(x.games, 'taflzero'), 10)} nodes per move "
        f"({num(median_nodes(m['s18000-t1000'].games, 'taflzero'), 10)} "
        f"at step 18,000; ours ~{num(median_nodes(x.games, OURS), 10)}).",
        match=x.name,
    )

    controls = (MATCHES / "paused_controls.txt").read_text(encoding="utf-8-sig")

    def in_controls(name: str) -> bool:
        """Whether paused_controls.txt has the match played between stopping and restarting training."""
        lines = [line.split(" ", 1)[1] for line in controls.splitlines() if line.strip()]
        try:
            i, j = lines.index(f"starting {name}"), lines.index(f"finished {name}")
            return lines.index("training stopped") < i < j < lines.index("restarting training")
        except ValueError:
            return False

    x = s26p
    s = x.summary
    wins_a, losses_a, _ = side_counts(x.games, "attackers")
    wins_d, losses_d, held_d = side_counts(x.games, "defenders")
    add(
        RESULTS,
        row_head(x, f", {consistent(in_controls(x.name), paused(x))} | {budget(x.header)} | ")
        + f"{x.header['pairs']} | {wl(s)} | {two(s['score'])}, {elo_with_interval(s)} ({s['pairs_won']} pairs won, "
        f"{s['pairs_lost']} lost, p = {p_text(s['sign_test_p'])}) |",
        consistent(
            x.header == {**s26.header} and [g["opening"] for g in x.games[:0]] == [],
            f"Same match without contention: TaflZero ~{num(median_nodes(x.games, 'taflzero'), 100)} nodes per move, "
            f"ours ~{num(median_nodes(x.games, OURS), 100)}.",
        ),
        f"As attacker: {wins_a.total()} wins ("
        + listing(wins_a, ["KingCaptured", "Repetition", "Encircled", "NoMoves"])
        + f"), {losses_a.total()} losses, {losses_a['ExitFort']} of them exit forts.",
        f"As defender: {listing(wins_d, ['KingEscaped'])}, {held_d.total()} held to 600 plies, "
        f"{losses_d['Repetition']} lost to repetition."
        if losses_d.total() == losses_d["Repetition"]
        else "As defender: [the records do not bear this out]",
        match=x.name,
    )

    x = m["s18000-t1000-paused"]
    s = x.summary
    wins_a, losses_a, _ = side_counts(x.games, "attackers")
    wins_d, losses_d, held_d = side_counts(x.games, "defenders")
    errors = (
        (MATCHES / x.record["runs"][-1]["errors"]).read_text(encoding="utf-8")
        if "errors" in x.record["runs"][-1]
        else ""
    )
    add(
        RESULTS,
        row_head(x, f", {consistent(in_controls(x.name), paused(x))} (control) | {budget(x.header)} |"),
        f"| {complete_pairs(x.games)} of {x.header['pairs']} | {wl(s)} |",
        f"| {wl(s)} | (match stopped after {s['games']} games) |",
        consistent(
            x.header["model"] == m["s18000-t1000"].header["model"], "The step-18,000 match replayed without contention"
        ),
        f"As attacker: {wins_a.total()} win, {losses_a['ExitFort']} exit-fort losses; as defender: all "
        f"{losses_d.total()} lost to repetition."
        if not wins_d and not held_d and losses_d.total() == losses_d["Repetition"]
        else "As attacker: [the records do not bear this out]",
        consistent(
            "engine exited" in errors and "_wait_for" in errors, "The arena stopped when TaflZero stopped answering"
        ),
        match=x.name,
    )

    x = one_s
    s = x.summary
    wins_a, losses_a, _ = side_counts(x.games, "attackers")
    wins_d, losses_d, held_d = side_counts(x.games, "defenders")
    rows = sorted(t.selfplay, key=lambda r: r["first_game"])
    stepped = [r for r in rows if r["approx_step"]]
    # The share of exit forts in the last 10,000 self-play games before the fort tactic (step 22,137)
    # and before step 36,000. Over 20,000 games the second is 10.8%; single blocks of 2,000 vary from 8%
    # to 17% around then.
    before = [r for r in stepped if int(r["approx_step"]) <= 22137][-5:]
    by_then = [r for r in stepped if int(r["approx_step"]) <= 36000][-5:]

    def fort_share(blocks: Sequence[dict[str, Any]]) -> float:
        return sum(r["ExitFort"] for r in blocks) / sum(r["games"] for r in blocks)

    add(
        RESULTS,
        row_head(x, f" ({millions(t.samples(36000))} samples, ~{num(t.hours(36000))} h), {paused(x)} | ")
        + f"{budget(x.header)} | {x.header['pairs']} | {wl(s)} | {two(s['score'])}, {elo_with_interval(s)} "
        f"({s['pairs_won']} pairs won, {s['pairs_lost']} lost, p = {p_text(s['sign_test_p'])}) |",
        f"TaflZero ~{num(median_nodes(x.games, 'taflzero'), 10)} nodes per move, "
        f"ours ~{num(median_nodes(x.games, OURS), 100)}.",
        f"As attacker: {wins_a.total()} wins ("
        + listing(wins_a, ["KingCaptured", "Repetition", "Encircled", "NoMoves"])
        + f"), {losses_a.total()} losses ({listing(losses_a, ['KingEscaped', 'NoMoves'])}), "
        + ("no exit forts" if not losses_a["ExitFort"] else f"{losses_a['ExitFort']} exit forts")
        + f" ({tally(as_side(s26p.games, 'attackers'), 'loss')['ExitFort']} at step 26,000).",
        f"As defender: {wins_d.total()} wins ("
        + listing(wins_d, ["KingEscaped", "ExitFort"], {"ExitFort": ("exit fort of our own", "exit forts of our own")})
        + f"), {held_d.total()} held to 600 plies, "
        + ("no losses." if not losses_d else f"{losses_d.total()} losses."),
        f"Forts had risen from ~{num(100 * fort_share(before))}% to ~{num(100 * fort_share(by_then))}% of "
        "self-play games",
        consistent(
            any(msg.startswith("start: step 22710,") for _, msg in t.events) and t.config["fort_threats"],
            "(step 22,710)",
        ),
        match=x.name,
    )

    # The 15 s match: its first run, then as it counts.
    x = fifteen
    first = first_run.games
    first_summary = summarize(first, OURS)
    final_rule = x.rule
    assert final_rule is not None
    under_rule = count_games(first, x.header, final_rule).counted
    rs = summarize(under_rule, OURS)
    wins_a, losses_a, _ = side_counts(first, "attackers")
    wins_d, losses_d, held_d = side_counts(first, "defenders")
    speeds = {n: speed(g, "taflzero") for n, g in enumerate(first, 1)}
    bands = [
        [n for n, v in speeds.items() if v >= 1400],
        [n for n, v in speeds.items() if 1000 <= v < 1400],
        [n for n, v in speeds.items() if 300 <= v < 1000],
        [n for n, v in speeds.items() if v < 300],
    ]
    not_first = [g for g in first if g not in under_rule]
    replayed = all(
        any(h["pair"] == g["pair"] and h["attackers"] == g["attackers"] for h in x.games[len(first) :])
        for g in not_first
    )
    add(
        RESULTS,
        f"| {dates([first_run])} | {RUN} step {step_of(first_run.header['model']):,}, {paused(x, [first_run])} | "
        f"{budget(first_run.header)}, one game at a time | {first_run.header['pairs']} | {wl(first_summary)}; "
        "under the "
        f"speed rule {wl(rs)} ({rs['games']} games) | {two(first_summary['score'])} over all {first_summary['games']} "
        f"games ({first_summary['pairs_won']} pairs won, p = {p_text(first_summary['sign_test_p'])}), but only "
        f"{rs['games']} games count |",
        f"First match at the target time ({first_run.header['leaves']} leaves per batch).",
        f"As attacker: {wins_a.total()} wins ("
        + listing(wins_a, ["KingCaptured", "Encircled", "NoMoves", "Repetition"])
        + f"). As defender: {wins_d['ExitFort']} exit forts, {held_d.total()} held to 600 plies."
        if not losses_a and not losses_d and wins_d.total() == wins_d["ExitFort"]
        else "As attacker: [the records do not bear this out]",
        f"TaflZero averaged {span((speeds[n] for n in bands[0]), 10)} nodes per second in games "
        f"{game_numbers(bands[0])} and {span(speeds[n] for n in bands[1])} in games {game_numbers(bands[1])}, but "
        f"{span((speeds[n] for n in bands[2]), 10)} in games {game_numbers(bands[2])}, and "
        f"{span((speeds[n] for n in bands[3]), 10)} in games {game_numbers(bands[3])}.",
        f"Games {game_numbers([n for n, g in enumerate(first, 1) if g in under_rule])} pass the speed rule "
        f"({rs['wins']} wins, {rs['unknown']} unfinished; "
        + ("both pairs won" if rs["pairs"] == rs["pairs_won"] == 2 else "[pairs]")
        + ").",
        consistent(
            replayed,
            f"The other {len(not_first)} did not count and were played again: "
            f"{sum(outcome(g) == 'win' for g in not_first)} of them had been wins, "
            f"{sum(outcome(g) == 'unfinished' for g in not_first)} unfinished.",
        ),
        match=x.name,
    )

    s = x.summary
    counted_replays = [g for g in x.counted if g not in first]
    replay_runs = [run for run in replays if any(g in counted_replays for g in run.games)]
    wins_a, losses_a, _ = side_counts(x.counted, "attackers")
    wins_d, losses_d, held_d = side_counts(x.counted, "defenders")
    lost_games = [g for g in as_side(x.counted, "attackers") if outcome(g) == "loss"]
    tz = [speed(g, "taflzero") for g in x.counted]
    stricter = [(v, outcome(g)) for v, g in zip(tz, x.counted, strict=True)]
    at_1500 = Counter(o for v, o in stricter if v >= 1500)
    at_1680 = Counter(o for v, o in stricter if v >= 1680)
    aside = Counter(why for _, why in x.counting.aside)
    first_aside = [g for g, _ in x.counting.aside if g in first]
    replay_aside = [(g, why) for g, why in x.counting.aside if g not in first]
    no_searches = [n for n, g in enumerate(x.counted, 1) if "taflzero_searches" not in g]
    start = min(run.created for run in replay_runs)
    start = start.replace(minute=5 * round(start.minute / 5), second=0)
    add(
        RESULTS,
        f"| {dates(x.runs)} | {RUN} step {step_of(x.header['model']):,}, {paused(x)} | {budget(x.header)}, "
        "one game at a "
        f"time, speed rule ({num(final_rule.nodes_per_s)} nodes per second) | {x.header['pairs']} | {wl(s)} | "
        f"{two(s['score'])}, {elo_with_interval(s)} ({s['pairs_won']} pairs won, "
        f"{'none' if not s['pairs_lost'] else s['pairs_lost']} lost, {s['pairs'] - s['pairs_won'] - s['pairs_lost']} "
        f"tied, p = {p_text(s['sign_test_p'])}) |",
        consistent(
            all(g in first for g in x.counted[: len(x.counted) - len(counted_replays)])
            and all(run.header.get("above_normal") for run in replay_runs)
            and not any(g.get("busy") for g in x.counted),
            f"games {game_numbers([n for n, g in enumerate(first, 1) if g in x.counted])} of the first run and "
            f"{len(counted_replays)} games replayed between {when(start)} and "
            f"{when(replay_runs[-1].last_written)}, at above normal priority, at times when no other process was on "
            "the GPU.",
        ),
        f"As attacker: {wins_a.total()} wins ("
        + listing(
            wins_a,
            ["NoMoves", "Encircled", "Repetition"],
            {"NoMoves": ("by leaving the defenders without a move",) * 2},
        )
        + f") and {losses_a.total()} loss (the king escaped after {lost_games[0]['plies']} plies)."
        if len(lost_games) == 1 and lost_games[0]["reason"] == "KingEscaped"
        else "As attacker: [the records do not bear this out]",
        f"As defender: {wins_d.total()} wins ({listing(wins_d, ['ExitFort', 'KingEscaped'])}) and {held_d.total()} "
        "games held to 600 plies."
        if not losses_d
        else "As defender: [the records do not bear this out]",
        f"TaflZero averaged {span(tz, sep=' to ')} nodes per second in these games (median "
        f"{num(statistics.median(tz), 10)}) and used "
        f"{span((g['taflzero_mean_ms'] / 1000 for g in x.counted), 0.1, ' to ')} s "
        f"per move; our engine used {span((g['ours_mean_ms'] / 1000 for g in x.counted), 0.1, ' to ')} s.",
        f"of the {at_1500.total()} in which TaflZero averaged 1,500 or more, {at_1500['win']} were won, "
        f"{at_1500['loss']} lost and {at_1500['unfinished']} unfinished; "
        + (f"all {at_1680.total()} at 1,680 or more were won." if at_1680.total() == at_1680["win"] else "[1,680]"),
        f"Caveats: {s['games']} games are a small sample.",
        f"before a game TaflZero reached between {num(min(all_quiet), 10)} and {num(max(all_quiet), 10)} nodes per "
        "second on the test position",
        f"Games {game_numbers(no_searches)} were recorded before single searches were logged, so the dip rule could "
        "not be applied to them.",
        f"Besides the {sum(1 for _, why in x.counting.aside if why == 'slow' and _ in first)} slow games of the first "
        f"run, {len(replay_aside)} replay games do not count: {aside['busy']} were stopped when another process took "
        f"the GPU or TaflZero's speed dipped, and {aside['dipped']} "
        + ("win was" if [outcome(g) for g in dips] == ["win"] else "[dipped]")
        + " set aside by the dip rule.",
        match=x.name,
    )
    if len(first_aside) != aside["slow"]:
        add(RESULTS, "[a replay game was set aside as slow]", match=x.name)

    # The match at step 468,000.
    x = m["s468000-t15000-paused"]
    s = x.summary
    step = step_of(x.header["model"])
    wins_a, losses_a, held_a = side_counts(x.counted, "attackers")
    wins_d, losses_d, held_d = side_counts(x.counted, "defenders")
    encircled = [g["plies"] for g in as_side(x.counted, "attackers") if g["reason"] == "Encircled"]
    others = [g["plies"] for g in as_side(x.counted, "attackers") if g["reason"] != "Encircled"]
    forts = [g["plies"] for g in as_side(x.counted, "defenders") if g["reason"] == "ExitFort"]
    tz = [speed(g, "taflzero") for g in x.counted]
    ours = [speed(g, OURS) for g in x.counted]
    fast = Counter(outcome(g) for g in x.counted if speed(g, "taflzero") >= 2400)
    tests = [q for run in x.runs for q, _ in run.quiet()]  # the speed test before each game
    tests_before = [q for run in fifteen.runs[1:] for q, _ in run.quiet()]
    stopped = [g for g, why in x.counting.aside if why == "busy"]
    by_run = [sum(1 for g in run.games if g in stopped) for run in x.runs]
    seventh = next(g for g in x.games if g["pair"] == 6)  # pairs are numbered from 0
    restarted = [g for run in x.runs[1:] for g in run.games if g in stopped]
    ladder = m["ladder-s468000-vs-s36000-n800"]
    low, _ = s["elo_ci95"]
    add(
        RESULTS,
        f"| {dates(x.runs)} | {RUN} step {step:,} ({millions(t.samples(step))} samples, {days} days), {paused(x)} | "
        f"{budget(x.header)}, one game at a time, speed rule ({num(x.rule.nodes_per_s if x.rule else 0)} nodes per "
        f"second) | {x.header['pairs']} | {wl(s)} | {two(s['score'])}, Elo {elo_text(s['elo'])} "
        + consistent(s["score_ci95"][1] >= 1, f"[{elo_text(low)} or more]")
        + f" ({s['pairs_won']} pairs won, {'none' if not s['pairs_lost'] else s['pairs_lost']} lost, "
        f"p = {p_text(s['sign_test_p'])}) |",
        consistent(same_setup(x, fifteen), "Same openings, rules and settings as the step-36,000 match;")
        + f" this network beat that one {wl(ladder.summary)} at {ladder.header['ours']}.",
        consistent(
            not losses_a and not held_a and set(wins_a) <= {"KingCaptured", "Encircled", "NoMoves"},
            f"As attacker: {wins_a.total()} wins ({wins_a['KingCaptured']} king captures, {wins_a['Encircled']} "
            f"encirclement after {'/'.join(map(str, encircled))} plies, {wins_a['NoMoves']} by leaving the "
            "defenders without a move);",
        )
        + f" apart from the encirclement they took {span(others, sep=' to ')} plies.",
        consistent(
            not losses_d and set(wins_d) == {"ExitFort"},
            f"As defender: {wins_d['ExitFort']} exit forts in {span(forts)} plies, {held_d.total()} games held to "
            "600 plies.",
        ),
        f"TaflZero averaged {span(tz, sep=' to ')} nodes per second (median {num(statistics.median(tz), 10)}), "
        + consistent(
            statistics.median(tz) > statistics.median(speed(g, "taflzero") for g in fifteen.counted)
            and statistics.median(tests) > statistics.median(tests_before),
            "faster than in the step-36,000 match because the machine was quieter:",
        )
        + f" on the test position before a game it reached {num(min(tests), 10)} to {num(max(tests), 10)} nodes "
        f"per second, against {num(min(tests_before), 10)} to {num(max(tests_before), 10)} before the replays of "
        f"that match. It used {span((g['taflzero_mean_ms'] / 1000 for g in x.counted), 0.1, ' to ')} s per move; "
        f"our engine used {span((g['ours_mean_ms'] / 1000 for g in x.counted), 0.1, ' to ')} s.",
        consistent(not fast["loss"], "Under a stricter limit the result holds too:")
        + f" in the {fast.total()} games in which TaflZero averaged 2,400 or more, our engine won {fast['win']}, "
        f"lost {fast['loss'] or 'none'} and left {fast['unfinished']} unfinished.",
        f"Our engine ran {span(ours, 100, ' to ')} simulations per second (median "
        f"{num(statistics.median(ours), 100)});",
        consistent(
            len(x.runs) == 3
            and not by_run[2]
            and x.rule is not None
            and not any(x.rule.dipped(g.get("taflzero_searches", ())) for g in stopped),
            f"{word(len(stopped)).capitalize()} games were stopped when another process took the GPU, and played "
            f"again: {word(by_run[0])} before the restart and {word(len(restarted))} after it,",
        ),
        consistent(
            all(g["pair"] == seventh["pair"] and g["attackers"] == seventh["attackers"] for g in restarted)
            and x.runs[1].games[0]["pair"] == seventh["pair"],
            "all of them the first game of the seventh pair.",
        ),
        consistent(
            s["score_ci95"][1] >= 1,
            f"The interval is a normal approximation over the {s['pairs']} pair scores, a rough one here; its upper "
            "end is above a score of 1, which has no Elo equivalent, so only the lower end is given.",
        ),
        match=x.name,
    )
    add(
        README,
        f"in the step-{step:,} match, the first in which the two engines count their search "
        f"the same way, this engine averaged {span(ours, 100, ' to ')} nodes per second over a game and TaflZero "
        f"{span(tz, sep=' to ')}.",
        match=x.name,
    )

    # Between our own checkpoints. Training was stopped, and restarted with a new learning rate.
    lr = [(int(r["step"]), r["lr"]) for r in t.metrics if r["step"] >= 500]  # after the warm-up
    change_step = next(step for step, rate in lr if rate != lr[0][1])
    restart = max(
        step
        for _, msg in t.events
        if (found := re.match(r"start: step (\d+),", msg)) and (step := int(found[1])) < change_step
    )
    stop_time, _ = t.event(rf"stopped at step {restart}")
    _, samples = t.started(restart)
    ladders = [m[name] for name in m if name.startswith("ladder-")]
    constant = {rate for step, rate in lr if step <= restart}
    after, before = m["s36000-t15000-paused"].runs[-1].last_written, m["s468000-t15000-paused"].runs[0].created
    add(
        RESULTS,
        consistent(after < stop_time < before, "Training went on after the step-36,000 match at 15 s,"),
        f"to step {restart:,} on {stop_time.date()} ({millions(samples)} samples, "
        f"{millions(t.at_or_before(restart)['games'])} self-play games).",
        consistent(
            all(y.header["ours"] == y.header["theirs"] == "800 nodes" for y in ladders)
            and all(y.header["leaves"] == 8 and y.header["pairs"] == 16 for y in ladders)
            and all(t.logged_during(y.runs) for y in ladders),
            "800 nodes per move for both sides (8 leaves per batch), 16 pairs, training running alongside,",
        ),
        "Progress had stopped at the constant learning rate of "
        f"{constant.pop() if len(constant) == 1 else '[varied]'}, "
        f"as the training loss also showed ({t.row(300000)['loss']:.2f} at step 300,000, "
        f"{t.row(469500)['loss']:.2f} at step 469,500). From step {restart:,} the learning rate is "
        f"{t.config['lr']}.",
    )
    if not all(rate == t.config["lr"] for step, rate in lr if step > restart):
        add(RESULTS, "[the learning rate after the change]")

    def ladder_row(name: str, tail: str) -> None:
        y = m[name]
        s = y.summary
        add(
            RESULTS,
            f"| {dates(y.runs)} | step {step_of(y.header['model']):,} against step {step_of(y.header['opponent']):,} | "
            f"{wl(s)} | {two(s['score'])}"
            + tail.format_map({**s, "elo": elo_text(s["elo"]), "p": p_text(s["sign_test_p"])}),
        )

    ladder_row("ladder-s468000-vs-s36000-n800", " ({pairs_won} pairs won, p = {p}) |")
    ladder_row("ladder-s148000-vs-s36000-n800", ", Elo {elo} ({pairs_won} pairs won, none lost, p = {p}) | |")
    ladder_row("ladder-s300000-vs-s148000-n800", ", Elo {elo} ({pairs_won} pairs won, {pairs_lost} lost, p = {p}) |")
    ladder_row("ladder-s468000-vs-s300000-n800", " ({pairs_won} pairs won, {pairs_lost} lost) |")
    y = m["ladder-s468000-vs-s36000-n800"]
    y3 = m["ladder-s300000-vs-s148000-n800"]
    y4 = m["ladder-s468000-vs-s300000-n800"]
    decided = [g for g in y4.games if g["winner"] is not None]
    add(
        RESULTS,
        consistent(
            y.summary["losses"] == 0 and y.summary["unknown"] == 0 and step_of(y.header["opponent"]) == 36000,
            "The network of step 36,000 lost every game, as attacker and as defender;",
        )
        + f" games took {num(statistics.mean(g['plies'] for g in y.games))} plies on average.",
        consistent(y3.summary["sign_test_p"] >= 0.05, "| Not significant. |"),
        consistent(
            y4.summary["pairs_won"] <= y4.summary["pairs_lost"],
            f"No progress in {step_of(y4.header['model']) - step_of(y4.header['opponent']):,} steps.",
        )
        + f" Between these two the attackers won {sum(g['winner'] == 'attackers' for g in decided)} of the "
        f"{len(decided)} decided games.",
    )
    return out


# --- Reading the documents ---


def visible(file: str) -> str:
    """A document without its HTML comments, its white space collapsed."""
    text = (ROOT / file).read_text(encoding="utf-8")
    return " ".join(re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).split())


def present(fragment: str, text: str) -> bool:
    return " ".join(fragment.split()) in text


def report(matches: dict[str, Match], t: Training) -> None:
    """What the records give, match by match."""
    for x in matches.values():
        s = x.summary
        state = " (in progress when copied; PENDING)" if x.name in PENDING else ""
        print(f"\n{x.name}{state}: {budget(x.header)}, {dates(x.runs)}, model {x.header['model']}")
        line = f"  counted {s['games']} of {len(x.games)} records: {wl(s)}"
        if "score" in s:
            line += f", score {s['score']:.4f}, {elo_with_interval(s)}, pairs {s['pairs_won']} won, "
            line += f"{s['pairs_lost']} lost of {s['pairs']}, sign test p = {s['sign_test_p']:.3g}"
        print(line)
        for side in ("attackers", "defenders"):
            games = as_side(x.counted, side)
            parts = [f"{result} {dict(tally(games, result))}" for result in ("win", "loss", "unfinished")]
            print(f"  as {side[:-1]}: " + "; ".join(parts))
        if x.counted:
            plies = [g["plies"] for g in x.counted]
            print(f"  plies {min(plies)}-{max(plies)}, mean {statistics.mean(plies):.1f}")
            for player in (OURS, x.opponent):
                speeds = [speed(g, player) for g in x.counted]
                seconds = [g[f"{player}_mean_ms"] / 1000 for g in x.counted]
                print(
                    f"  {player}: nodes per move median {median_nodes(x.counted, player):.0f}; nodes/s "
                    f"{min(speeds):.0f}-{max(speeds):.0f} (median {statistics.median(speeds):.0f}); "
                    f"s per move {min(seconds):.2f}-{max(seconds):.2f}"
                )
        if x.counting.aside:
            print(f"  not counted: {dict(Counter(why for _, why in x.counting.aside))}")
        quiet = [s for run in x.runs for s, _ in run.quiet()]
        if quiet:
            print(f"  test speeds before games: {min(quiet)}-{max(quiet)} ({len(quiet)} readings)")
    fifteen = matches["s36000-t15000-paused"]
    slowdown = first_search_slowdown(fifteen)
    print(
        "\nfirst search of a replay game, busier half against quieter half by the test speed before it: "
        + ", ".join(f"{player} {100 * v:.0f}% slower" for player, v in slowdown.items())
    )
    print("\ntraining:")
    for step in (1699, 6000, 10000, 16000, 18000, 26000, 36000, 148000, 300000, 468000):
        print(f"  step {step:,}: {t.samples(step):,} samples, {t.hours(step):.2f} h after the first logged step")


# --- The games as data (--json) ---

# Why a record does not count, as the arena decides when it resumes a match from its records.
WHY = {
    "busy": "busy",  # another process was on the GPU while it was played (the record's busy flag)
    "dipped": "dip",  # five of TaflZero's searches in a row averaged under 3/4 of the limit
    "slow": "slow",  # TaflZero averaged less than the limit in force when it was played
    # (and "limit-raise": TaflZero averaged an earlier, lower limit of the match but not the final one;
    # "superseded": its game had already counted, or been given up, when it was played)
}
JSON_NOTE = """\
Every game of every match in matches/, in the order played, and what counts, from \
tools/check_results.py --json. Games: index is the line of the match's .jsonl from 0 (docs/RESULTS.md \
numbers games from 1); run is the arena run from 0, log its log in matches/; ours is the side our engine \
played; result is ours (win, loss or unfinished); ours_ and theirs_nodes_per_s are the engines' mean \
speeds in the game (mean nodes per move over mean time per move); searches_recorded is whether the record \
has the single searches, without which the dip rule cannot be applied; limit is the speed rule's limit \
for TaflZero that decides whether the game counts (that of the match's last run), limit_when_played the \
one in force when it was played (null: none). why, for a game that does not count: busy (another process \
was on the GPU), dip (five TaflZero searches in a row under 3/4 of the limit), slow (TaflZero under the \
limit in force when it was played), limit-raise (TaflZero over an earlier, lower limit of the match but \
under the final one), superseded (its game had already counted or been given up). Matches: counted \
holds the results of \
the counted games for our engine, as docs/RESULTS.md gives them: pairs are the complete pairs; score is \
the mean pair score with a 95% interval, elo its Elo difference (clamped at +-2400 for a score of 0 or 1) \
and sign_test_p the two-sided sign test over pairs won and lost; text holds them as written there. \
first_run, for a match whose first run had another limit, is the first run's games counted as they \
were then. pending marks a match that was still being played when its records were copied."""


def totals(games: list[dict[str, Any]]) -> dict[str, Any]:
    """The results of games for our engine, as the documents give them."""
    s = summarize(games, OURS)
    sides = {}
    for side in ("attackers", "defenders"):
        played = as_side(games, side)
        sides[f"as_{side}"] = {
            "games": len(played),
            "wins": dict(tally(played, "win")),
            "losses": dict(tally(played, "loss")),
            "unfinished": dict(tally(played, "unfinished")),
        }
    won, lost = s.get("pairs_won", 0), s.get("pairs_lost", 0)
    scored = "score" in s
    return {
        "games": s["games"],
        "wins": s["wins"],
        "losses": s["losses"],
        "unfinished": s["unknown"],
        **sides,
        "pairs": s["pairs"],
        "pairs_won": won,
        "pairs_tied": s["pairs"] - won - lost,
        "pairs_lost": lost,
        "score": s.get("score"),
        "score_ci95": list(s["score_ci95"]) if scored else None,
        "elo": s.get("elo"),
        "elo_ci95": list(s["elo_ci95"]) if scored else None,
        "sign_test_p": s.get("sign_test_p"),
        "text": {
            "result": wl(s),
            "score": two(s["score"]) if scored else None,
            "elo": elo_with_interval(s) if scored else None,
            "p": p_text(s["sign_test_p"]) if scored else None,
        },
    }


def games_data(x: Match) -> list[dict[str, Any]]:
    """Every record of a match, in the order played, with whether it counts and why not."""
    status: dict[int, tuple[bool, str | None]] = {id(g): (True, None) for g in x.counted}
    status.update({id(g): (False, WHY[why]) for g, why in x.counting.aside})
    status.update({id(g): (False, "superseded") for g in x.counting.unused})
    final = x.rule.nodes_per_s if x.rule else None
    lowest = min((run.header.get("min_opponent_speed") or math.inf for run in x.runs), default=math.inf)
    out, index = [], 0
    for number, run in enumerate(x.runs):
        played_under = run.header.get("min_opponent_speed") or None
        for g in run.games:
            counted, why = status[id(g)]
            theirs = speed(g, x.opponent)
            if why == "slow" and final and lowest <= theirs < final and (played_under or 0) < final:
                why = "limit-raise"
            out.append(
                {
                    "index": index,
                    "run": number,
                    "log": run.log,
                    "pair": g["pair"],
                    "opening": g["opening"],
                    "ours": "attackers" if g["attackers"] == OURS else "defenders",
                    "winner": g["winner"],
                    "reason": g["reason"],
                    "plies": g["plies"],
                    "result": outcome(g),
                    "ours_nodes_per_s": round(speed(g, OURS), 1),
                    "theirs_nodes_per_s": round(theirs, 1),
                    "searches_recorded": f"{x.opponent}_searches" in g,
                    "counted": counted,
                    "why": why,
                    "limit": final,
                    "limit_when_played": played_under,
                }
            )
            index += 1
    return out


def games_json(matches: dict[str, Match]) -> str:
    """The --json file: the games of every match and its counted results; one game per line."""
    entries, games = [], []
    for x in sorted(matches.values(), key=lambda x: x.runs[0].created):
        data = games_data(x)
        entry = {
            "match": x.name,
            "pending": x.name in PENDING,
            "model": x.header["model"],
            "opponent": x.header["opponent"],
            "budget": budget(x.header),
            "pairs_scheduled": x.header["pairs"],
            "dates": dates(x.runs),
            "start": x.runs[0].created.isoformat(),
            "end": x.runs[-1].last_written.isoformat(),
            "limit": x.rule.nodes_per_s if x.rule else None,
            "runs": [
                {
                    "log": run.log,
                    "start": run.created.isoformat(),
                    "end": run.last_written.isoformat(),
                    "limit": run.header.get("min_opponent_speed") or None,
                    "games": len(run.games),
                }
                for run in x.runs
            ],
            "counted": totals(x.counted),
            "not_counted": dict(Counter(g["why"] for g in data if not g["counted"])),
            "games": f"@games{len(games)}@",
        }
        if (x.runs[0].header.get("min_opponent_speed") or None) != entry["limit"]:
            entry["first_run"] = totals(x.runs[0].games)
        entries.append(entry)
        games.append(data)
    text = json.dumps({"note": JSON_NOTE, "matches": entries}, indent=2, ensure_ascii=False)
    for i, data in enumerate(games):
        rows = ",\n".join(f"        {json.dumps(g, ensure_ascii=False)}" for g in data)
        text = text.replace(f'"@games{i}@"', f"[\n{rows}\n      ]")
    return text + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="check every stated number against the records")
    parser.add_argument(
        "--json",
        type=Path,
        metavar="PATH",
        help="write every game of every match, in the order played, with whether it counts and why not, and "
        "each match's counted results, to PATH",
    )
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(errors="backslashreplace")  # type: ignore[union-attr]
    names = [p.stem for p in sorted(MATCHES.glob("*.json"))]
    matches = {name: load_match(name) for name in names}
    zones = {run.created.utcoffset() for x in matches.values() for run in x.runs}
    if len(zones) != 1:
        raise SystemExit("check_results: the match records are in more than one time zone")
    training = load_training(timezone(zones.pop()))  # type: ignore[arg-type]
    if args.json:
        args.json.write_text(games_json(matches), encoding="utf-8", newline="\n")
        print(f"wrote {args.json}")
        if not args.check:
            return 0
    if not args.check:
        report(matches, training)
        return 0
    status = 0
    for problem in check_records(matches, training):
        print(f"DIFFERS: {problem}")
        status = 1
    fragments = stated(matches, training)
    texts = {file: visible(file) for file in (README, RESULTS, CHANGELOG)}
    pending = [f for f in fragments if f.match in PENDING]
    for file, text in texts.items():
        missing = [f for f in fragments if f.file == file and f.match not in PENDING and not present(f.text, text)]
        unlisted = [u for u in NOT_CHECKED if u.file == file and not present(u.text, text)]
        for f in missing:
            print(f"MISSING in {file}: {' '.join(f.text.split())}")
        for u in unlisted:
            print(f"MISSING in {file} (listed as not checked): {u.text}")
        if missing or unlisted:
            status = 1
        else:
            print(f"agrees with the records: {file}")
    for name in sorted(PENDING):
        mine = [f for f in pending if f.match == name]
        absent = [f for f in mine if not present(f.text, texts[f.file])]
        print(f"pending, not failing: {name}: {len(mine) - len(absent)} of {len(mine)} fragments found")
        for f in absent:
            print(f"  pending, not in {f.file}: {' '.join(f.text.split())}")
    for u in NOT_CHECKED:
        print(f"not checked: {u.text} ({u.why})")
    for entry in manifest():
        if not (MODELS / entry["file"]).exists():
            print(
                f"not checked: models/{entry['file']}, against its size, SHA-256 and counters in "
                "models/manifest.json and the network the code describes (the checkpoints are not "
                "distributed; put a copy in models/ to check it)"
            )
    return status


if __name__ == "__main__":
    sys.exit(main())
