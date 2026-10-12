"""Self-play trends of a training run, per block of games.

python tools/run_stats.py runs/r1 [--block 2000]
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections import Counter
from pathlib import Path

REASONS = ["KingEscaped", "ExitFort", "KingCaptured", "Encircled", "Repetition", "NoMoves"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--block", type=int, default=2000)
    args = parser.parse_args()
    games = []
    with open(args.run / "selfplay" / "games.jsonl", encoding="utf-8", errors="replace") as f:
        for line in f:
            with contextlib.suppress(json.JSONDecodeError):  # lines torn by a crash
                games.append(json.loads(line))
    header = f"{'games':>13} {'att win':>8} {'unknown':>8} {'plies':>6} " + " ".join(f"{r[:9]:>9}" for r in REASONS)
    print(header)
    for start in range(0, len(games), args.block):
        block = games[start : start + args.block]
        winners = Counter(g["winner"] for g in block)
        reasons = Counter(g["reason"] for g in block)
        decided = winners["attackers"] + winners["defenders"]
        att = winners["attackers"] / decided if decided else 0.0
        plies = sum(g["plies"] for g in block) / len(block)
        shares = " ".join(f"{reasons[r] / len(block):>9.3f}" for r in REASONS)
        span = f"{start + 1}-{start + len(block)}"
        print(f"{span:>13} {att:>8.3f} {winners[None] / len(block):>8.3f} {plies:>6.0f} {shares}")


if __name__ == "__main__":
    main()
