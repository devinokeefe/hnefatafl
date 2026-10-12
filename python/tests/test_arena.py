import itertools
import json
from collections.abc import Collection

from tafl.arena import INTERRUPTED, GpuWatch, MoveResult, SpeedRule, play_match, random_openings
from tafl.native import game_info


class Scripted:
    """Plays the first legal move at once and reports a search speed chosen per game."""

    def __init__(self, name: str, slow_games: Collection[int] = ()) -> None:
        self.name = name
        self.slow_games = slow_games
        self.asked: list[tuple[int, list[str]]] = []

    def request(self, game: int, moves: list[str]) -> None:
        self.asked.append((game, list(moves)))

    def step(self) -> list[tuple[int, MoveResult]]:
        out = []
        for game, moves in self.asked:
            legal = game_info("startpos", " ".join(moves))["legal"]
            nodes = 500 if game in self.slow_games else 2000
            out.append((game, MoveResult(legal[0], nodes, 1000.0)))
        self.asked = []
        return out

    def close(self) -> None:
        pass


def test_slow_games_are_set_aside_and_played_again(tmp_path):
    openings = random_openings(1, 2, 1)
    log = tmp_path / "match.jsonl"
    rule = SpeedRule("b", 1000)
    # The first game started (number 0) is the slow one.
    games, set_aside = play_match(Scripted("a"), Scripted("b", {0}), openings, 1, 8, log, rule=rule)
    assert set_aside == 1
    assert [(g["attackers"], g["reason"]) for g in games] == [("a", "PlyCap"), ("b", "PlyCap")]
    assert all(rule.counts(g) for g in games)
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 3 and not rule.counts(records[0])

    # Resuming from the record plays nothing more and reaches the same result.
    again, set_aside = play_match(Scripted("a"), Scripted("b"), openings, 1, 8, None, records, rule)
    assert again == games and set_aside == 1

    # A game that stays slow is given up after its last attempt.
    stubborn = SpeedRule("b", 1000, attempts=2)
    games, set_aside = play_match(Scripted("a"), Scripted("b", {0, 1}), openings, 1, 8, None, rule=stubborn)
    assert set_aside == 2 and [g["attackers"] for g in games] == ["b"]


def test_before_game_runs_ahead_of_every_game():
    calls = []
    openings = random_openings(2, 2, 1)
    play_match(Scripted("a"), Scripted("b"), openings, 1, 6, None, before_game=lambda: calls.append(1))
    assert len(calls) == 4


class Watch:
    """Reports the GPU as taken on the numbered calls of each kind given."""

    def __init__(self, taken: Collection[int] = (), taken_now: Collection[int] = ()) -> None:
        self.answers = {"taken": taken, "taken_now": taken_now}
        self.calls = {"taken": 0, "taken_now": 0}

    def ask(self, kind: str) -> bool:
        self.calls[kind] += 1
        return self.calls[kind] in self.answers[kind]

    def taken(self) -> bool:
        return self.ask("taken")

    def taken_now(self) -> bool:
        return self.ask("taken_now")


def test_games_stop_when_another_process_takes_the_gpu(tmp_path):
    openings = random_openings(1, 2, 1)
    log = tmp_path / "match.jsonl"
    rule = SpeedRule("b", 1000, attempts=1)
    # Taken while the first game is being played: it stops at its next move.
    games, set_aside = play_match(Scripted("a"), Scripted("b"), openings, 1, 8, log, rule=rule, watch=Watch({1}))
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert (records[0]["reason"], records[0]["plies"], records[0]["busy"]) == (INTERRUPTED, 3, True)
    assert set_aside == 1 and [g["attackers"] for g in games] == ["a", "b"]
    assert not any(g.get("busy") for g in games)

    # Taken just before the first game ended: its result stands in the record but does
    # not count, and it is played again however often that happens.
    games, set_aside = play_match(
        Scripted("a"), Scripted("b"), openings, 1, 8, log, rule=rule, watch=Watch(taken_now={1, 2})
    )
    records = [json.loads(line) for line in log.read_text().splitlines()][3:]
    assert [(r["reason"], r.get("busy", False)) for r in records] == [("PlyCap", True)] * 2 + [("PlyCap", False)] * 2
    assert set_aside == 2 and len(games) == 2


def test_a_dip_in_speed_stops_the_game(tmp_path):
    rule = SpeedRule("b", 1000)  # a dip is five searches in a row below 750 nodes per second
    fast, slow, at_once = [2000, 1000.0], [500, 1000.0], [0, 2.0]
    assert not rule.dipped([fast] * 20)
    assert not rule.dipped([fast, slow, slow, fast, fast, fast])
    assert rule.dipped([*[fast] * 8, *[slow] * 5, *[fast] * 8])
    assert not rule.dipped([at_once] * 9)  # forced replies say nothing about speed
    assert rule.dipped([slow, at_once, slow, slow, at_once, slow, slow])

    # In a long enough game the slow engine is caught after its fifth search.
    openings = random_openings(1, 2, 1)
    log = tmp_path / "match.jsonl"
    games, set_aside = play_match(Scripted("a"), Scripted("b", {0}), openings, 1, 14, log, rule=rule)
    first = json.loads(log.read_text().splitlines()[0])
    assert (first["reason"], first["plies"], len(first["b_searches"])) == (INTERRUPTED, 12, 5)
    assert not rule.counts(first) and set_aside == 1
    assert [(g["attackers"], g["reason"]) for g in games] == [("a", "PlyCap"), ("b", "PlyCap")]


def test_gpu_watch_needs_two_high_readings_in_a_row():
    def watch(loads: list[float]) -> GpuWatch:
        endless = itertools.cycle(loads)
        return GpuWatch({1}, limit=15.0, pause_s=0.01, read=lambda ours: next(endless))

    busy, quiet, blips = watch([70.0]), watch([5.0]), watch([80.0, 5.0])
    for w in (busy, quiet, blips):
        w.taken_now()  # by the next call there are two readings
    assert busy.taken_now() and busy.taken()
    assert not quiet.taken_now()
    assert not any(blips.taken_now() for _ in range(6))
