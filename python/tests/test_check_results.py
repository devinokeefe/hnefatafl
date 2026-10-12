"""tools/check_results.py repeats tafl.arena's rule for which games count, without PyTorch. These tests
hold the two to the same answers on every match in matches/."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tafl import arena

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_results", ROOT / "tools" / "check_results.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_results"] = module  # its dataclasses look themselves up there
    spec.loader.exec_module(module)
    return module


def matches(tool: ModuleType) -> list:
    return [tool.load_match(p.stem) for p in sorted((ROOT / "matches").glob("*.json"))]


def test_speed_rule_agrees(tool: ModuleType) -> None:
    for match in matches(tool):
        if match.rule is None:
            continue
        theirs = arena.SpeedRule(match.rule.player, match.rule.nodes_per_s, match.rule.attempts)
        for g in match.games:
            assert theirs.counts(g) == match.rule.counts(g), (match.name, g["pair"], g["attackers"])
            assert theirs.disturbed(g) == match.rule.disturbed(g), (match.name, g["pair"], g["attackers"])


def test_resuming_counts_the_same_games(tool: ModuleType) -> None:
    """play_match, resumed from all the records of a match that is complete, plays nothing and keeps
    the games that check_results counts."""
    checked = 0
    for match in matches(tool):
        if len(match.counted) < 2 * match.header["pairs"]:
            continue  # stopped or still being played: play_match would play the rest
        openings = [None] * match.header["pairs"]
        for g in match.games:
            openings[g["pair"]] = g["opening"].split()
        rule = match.rule and arena.SpeedRule(match.rule.player, match.rule.nodes_per_s, match.rule.attempts)
        players = [SimpleNamespace(name=tool.OURS), SimpleNamespace(name=match.opponent)]
        games, aside = arena.play_match(*players, openings, 1, 600, None, match.games, rule)
        assert [id(g) for g in games] == [id(g) for g in match.counted], match.name
        assert aside == len(match.counting.aside), match.name
        checked += 1
    assert checked >= 10


def test_summaries_agree(tool: ModuleType) -> None:
    for match in matches(tool):
        theirs = json.loads(json.dumps(arena.summarize(match.counted, tool.OURS)))
        mine = json.loads(json.dumps(tool.summarize(match.counted, tool.OURS)))
        assert theirs == mine, match.name
