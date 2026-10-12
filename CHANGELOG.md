# Changelog

## 0.1.0 (2026-10-12)

The first release.

* **Rules.** Copenhagen Hnefatafl on the 11×11 board, as TaflZero plays it ([RULES.md](RULES.md)),
  in Rust (`tafl-core`), checked position by position against TaflZero's own rules.
* **Engine.** Batched Monte Carlo tree search (Gumbel AlphaZero in self-play, PUCT in play) with exact
  short tactics, the `tafl` command and a Python package (`tafl`).
* **Training.** Self-play from random weights in one process on one GPU, with 10-block,
  128-channel residual networks. The networks of steps 36,000 and 468,000, which played the
  15-second matches, are identified by size and SHA-256 in `models/manifest.json`.
* **Matches.** The records of every match against TaflZero, among them two at 15 seconds per move, and of
  the matches between the project's own networks ([docs/RESULTS.md](docs/RESULTS.md), `matches/`).
  At 15 seconds per move the network of step 468,000 won all 8 opening pairs against TaflZero, with
  14 games won, none lost and 2 unfinished. `tools/check_results.py` recomputes every result in
  RESULTS.md, the README and this file from the records.
* **Checks.** `python tools/check_all.py`, and CI on Linux, macOS and Windows
  ([CONTRIBUTING.md](CONTRIBUTING.md)). The last full comparison of the rules with TaflZero's is
  recorded in [docs/difftest.txt](docs/difftest.txt).

The self-play games and training samples (over 60 GB) and the training checkpoints, those two
networks among them, are not in the repository.
