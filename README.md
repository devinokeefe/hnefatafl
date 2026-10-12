# Learning Hnefatafl

Hnefatafl is the board game the Vikings played before chess: a king and his defenders start in the
middle of the board and try to bring him to a corner, against twice as many attackers who try to
trap him. This repository holds an engine for Copenhagen Hnefatafl, the modern rules for the 11×11
board, that learned to play from the rules and from its games against itself, in the manner of
AlphaZero: a Rust rules and search core, and a PyTorch network trained only on the engine's own
games. Its search was also given a few exact tactics by hand ([How it works](#how-it-works)).

The aim was to beat TaflZero, an open-source engine of the same kind, running its network g0213, at
15 seconds a move for both engines (the time limit of the Fetlar world championship), on one
computer, one game at a time.

**At 15 seconds a move, the network of step 468,000, after 7.7 days of training, won 14 of 16 games
against TaflZero, lost none and left 2 unfinished at the 600-ply limit: all 8 opening pairs won
(sign test p = 0.008).** With the same openings, the network of step 36,000, after about 13 hours
of training, had won 12, lost 1 and left 3 unfinished: 7 pairs won, 1 tied and none lost
(p = 0.016). At 1 second a move that network had won 20 games, lost 3 and left 9 unfinished, with
13 of 16 pairs won and 2 lost.

Sixteen games a match are a small sample. The opponent is TaflZero as built on this computer, from
a modified copy of its source ([The opponent](#the-opponent)). And the engines are compared as
wholes, speed included: in the step-468,000 match, the first in which the two engines count their
search the same way, this engine averaged 6,400 to 33,800 nodes per second over a game and TaflZero
1,650 to 3,066. Every match is in [docs/RESULTS.md](docs/RESULTS.md), and their records are in
`matches/`.

## Play it

You need Rust 1.85 or newer, Python 3.10 or newer with PyTorch and NumPy, and a network. The
networks that played the matches are not distributed, so it has to be one you train
([Reproducing it](#reproducing-it)); training saves one every 2,000 steps in
`runs/<run>/checkpoints/`. From the repository's root:

```bash
cargo build --release
python -m pip install -e ".[dev]"
python -m tafl.play --model runs/r1/checkpoints/step_0036000.pt   # http://localhost:8080
```

This is the whole engine, network and search, at a time per move chosen on the page, from half a
second to ten; the matches gave it fifteen. It runs on the GPU if PyTorch finds one, and otherwise
on the processor, more slowly.

## Results

The matches against TaflZero at 15 seconds a move were played one game at a time, with training
paused. Each of 8 pairs plays a random two-ply opening twice, with the sides swapped, which evens
out the luck of the opening and of the side, and the statistics use the pair as the unit. A game
still running after 600 plies is unfinished and counts half a point to each side.

| Network | Trained | Games won, lost, unfinished | Pairs won, tied, lost | Sign test |
|---|---|---|---|---|
| step 36,000 | about 13 hours, 265,148 games | 12, 1, 3 | 7, 1, 0 | p = 0.016 |
| step 468,000 | 7.7 days, 2,589,127 games | 14, 0, 2 | 8, 0, 0 | p = 0.008 |

**The speed rule.** The computer is shared, and a busy computer slowed TaflZero far more than this
engine: in a match at 1 second a move with training running alongside, TaflZero searched about 280
nodes per move against about 600 with training paused. So a game counts only if TaflZero averaged
at least 1,350 nodes per second over its moves (meant as 80% of its speed on a quiet machine), no
other process kept the GPU busy during the game, and no five TaflZero searches of a second or more in
a row averaged less than three quarters of that limit. Before each game the referee waits until
TaflZero reaches that speed on a test position and the GPU is free. A game that fails stays in the
records and is played again, **whatever its result**; one that was merely slow is given up after
`--attempts` plays (3 by default), and no game was slow twice. In the step-36,000 match the rule was
introduced and tightened while the match was being played; every finished game it set aside had
been a win or unfinished, so it never took a loss away from this engine. For the step-468,000 match
it was fixed before the first game.

[docs/RESULTS.md](docs/RESULTS.md) has the history of the rule, every other match against TaflZero,
and the matches between the project's own networks, which found clear progress only up to step
148,000 and none after step 300,000, at a constant learning rate.
`python tools/check_results.py --check` recomputes every result in RESULTS.md, in CHANGELOG.md and
in this file from the records.

## The opponent

[TaflZero](https://github.com/sovelin/taflzero) (GPL-3.0) is an AlphaZero-style engine for 11×11
Copenhagen Hnefatafl, and the engine of [playtafl.org](https://playtafl.org). The build these
matches used is not in this repository, and it is not a stock release:

* It was built at commit `d5fdcd2` of an unpublished copy of TaflZero's source, made for an earlier
  project: upstream commit `9b5a19a` with 41 local commits. Most add tests, logging, the
  recording of training data, or options that are off unless set. A few change how it plays: a side
  with no legal move loses (as the Copenhagen rules say; upstream had no such check), and its search
  scores such positions as lost; its search tree is reused between moves only along the moves
  actually played; and the flag that decides when its search tests for encirclement is restored
  correctly when a move is taken back.
* It ran the network g0213 (`T2-10x128-g0213.onnx`), the one playtafl.org's page was loading when
  it was downloaded. No licence for that network was found, so it is not included either.
* No record compares the strength of this build with stock TaflZero.

To play matches of your own, put the paths of a TaflZero executable, its network and ONNX Runtime in
`taflzero.local.json` (see `TaflZeroSetup` in `python/tafl/arena.py`). The step-468,000 match was
started with these settings, `--model` naming the checkpoint of that step:

```bash
python -m tafl.arena --model runs/r1/checkpoints/step_0468000.pt --movetime 15000 --pairs 8 --leaves 16 --concurrency 1 --min-opponent-speed 1350 --above-normal --out runs/arena/15s.jsonl
```

and the last runs of the step-36,000 match used the same settings with the checkpoint of step
36,000. `--leaves 16` makes this engine's search collect 16 leaves, with
virtual loss, for each call to the network, and `--above-normal` runs both engines at above-normal
priority. A match interrupted for any reason continues with `--resume`. Matches at a fixed number of nodes per move (`--nodes 400`) compare the engines at equal
search: both keep their search tree between moves, and the budget counts only the nodes added for
the current move.

## Rules

[RULES.md](RULES.md) gives the rules exactly as the engine plays them: the Copenhagen rules as
TaflZero implements them, which settles the cases the official text leaves open. For example, a
position that occurs for the third time wins for the attackers, and exit forts and encirclement are
tested only after certain moves.

`tafl difftest` plays games of random moves through this engine's rules and TaflZero's and compares
every position, legal-move list, repetition count and result. Most of its games weight the moves
towards captures, game endings, the edge and the king; over half start from set-up positions, random
ones and ones built for shieldwalls and encirclements; and some keep undoing their own moves to bring
about repetitions. The last full run: 20,000 games, 4,091,433 positions, 0 mismatches. Its record is
[docs/difftest.txt](docs/difftest.txt). The comparison needs a research probe, a small program from an
earlier project that links the rules library of the TaflZero build above; it is not distributed
with this project. `tafl dump` writes the same kind of games, with every position's legal moves,
captures, repetition counts, results and network inputs, as a reference for other implementations
of the rules. Counting the move sequences from the start (perft) gives 116, 6,788 and 806,344 for
one, two and three plies.

## How it works

**Rules** (`crates/tafl-core`): 128-bit bitboards, sliding moves from a precomputed line table,
Zobrist hashing, the eight symmetries of the board.

**Search** (`crates/tafl-engine`):

* Gumbel AlphaZero (sequential halving with Gumbel noise, completed-Q policy improvement, as in
  DeepMind's mctx) in self-play, and PUCT with first-play urgency in matches. In self-play a pool
  runs thousands of searches at once and hands their leaves to the GPU together; in a match the one
  search collects a batch of leaves (`--leaves`) before each network call.
* Exact short tactics at every new node. Immediate king escapes, king captures, third repetitions
  and exit forts are taken, as are escapes in two king moves. When the king threatens an escape, in
  one move or two, or the defenders threaten an exit fort, the attackers search only the moves that
  stop it; when the king's capture is threatened, the defenders search only the moves that prevent
  it. Proven wins and losses propagate up the tree (MCTS-solver). The repetition and fort tactics
  were added after matches against TaflZero showed the search blind to them; the others were there
  from the first match.
* In self-play, defender moves that threaten to complete an exit fort get a first visit at the
  root, up to half of the 16 moves the root considers, since the network is slow to propose them.
* Repetition is tracked along the search path as well as the game, so the search sees third
  occurrences exactly.

**Network** (`python/tafl/model.py`): a 10-block, 128-channel residual tower with squeeze-excitation
(the same size as TaflZero's), a policy head over the 4,840 move actions (40 move planes over the
board), and value heads for the result, the search's value and the number of moves left. Inputs are
20 planes, including the last four moves and the destinations of moves that would repeat a position
for the second or third time. TaflZero's 11 input planes have no move history and, for repetition,
mark only how often the position itself has occurred. Inference runs a fused float16 copy, with the batch norms folded in.

**Training** (`python/tafl/train.py`): one process, with a self-play thread and a training thread
sharing the GPU. Choices from KataGo and related work: playout-cap randomisation (25% of moves get a
full search and become training samples), a replay window that grows with the data, an auxiliary
target for the opponent's reply, and the eight symmetries for augmentation. Early on the defenders
win almost every game, so every attacker move looks equally lost. To give the attackers something
to learn from, self-play search values are shaped by the predicted game length (a losing side
prefers to survive longer, a winning side to win sooner), as Leela Chess Zero's moves-left utility
does.

## Reproducing it

**The checks.** Formatting, lints, the Rust and Python tests, the results against the records and
the links between the documents. What needs something missing here (PyTorch, a GPU, the research
probe, the networks) is skipped and listed:

```bash
python tools/check_all.py
```

**The training.** The run behind every result here trained on one desktop computer, with an Intel
i9-14900K (32 threads), an NVIDIA RTX 4090 and 64 GB of memory: 7.7 days to step 468,000. Its
configuration, metrics, self-play statistics and log of events are in `training/`. On Windows it
runs detached under a supervisor that restarts it after a crash, with Python from a virtual
environment in `.venv`; create `runs/r1/STOP` to stop it, and run the same command to resume:

```bash
powershell -File tools/start_training.ps1 -Run runs/r1
```

On other systems run the supervisor itself, `python -m tafl.supervise runs/r1`, from the
repository's root. Self-play is not deterministic: thousands of games share the GPU in whatever
order they reach it. A second run will not reproduce the same networks. It will also have all the
search's tactics, and the first visits to fort threats (`fort_threats` in the configuration), from
its first step, where r1 gained several of them partway through.

**The matches** need TaflZero ([The opponent](#the-opponent)) and about ten hours each at 15 seconds
a move, more when games have to be played again.

## What is where

```
RULES.md             the rules, exactly as implemented
crates/tafl-core     rules, positions, games, symmetries
crates/tafl-engine   input encoding, tactics, MCTS, the self-play and analysis pools
crates/tafl-ffi      the C ABI that the Python package uses
crates/tafl-cli      the `tafl` command: perft, difftest, dump
python/tafl          bindings, network, training, the match referee (arena.py), browser play, export
models/              manifest.json: the two networks that played the 15-second matches, by size
                     and SHA-256 (the networks themselves are not distributed)
matches/             the records of every match in docs/RESULTS.md (matches/README.md)
training/            the training run's configuration, metrics, self-play statistics and events
tools/               check_all.py, check_results.py, check_release.py, and the scripts that copied
                     the records from the run's folders
docs/                RESULTS.md, the results; FORMATS.md, the network file for other runtimes;
                     difftest.txt, the record of the last full rules check against TaflZero
```

Not in the repository:

| What | Size | Why |
|---|---:|---|
| The training run's self-play games and training samples | over 60 GB | their size |
| The training checkpoints, one every 2,000 steps and one at each stop, among them the two networks that played the 15-second matches | 12.8 MB each | not distributed; `models/manifest.json` identifies those two by size and SHA-256 |
| The TaflZero build, its research probe and the g0213 network | | not this project's, and the build and probe are unpublished; see [The opponent](#the-opponent) |

## References, licence, citing

D. Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go
through self-play", *Science* 362 (2018), 1140–1144. I. Danihelka, A. Guez, J. Schrittwieser and D.
Silver, "Policy improvement by planning with Gumbel", ICLR 2022, and DeepMind's mctx library,
https://github.com/google-deepmind/mctx. D. J. Wu, "Accelerating self-play learning in Go",
arXiv:1902.10565 (2019). H. Forstén, "Moves left head", Leela Chess Zero pull request 961,
https://github.com/LeelaChessZero/lc0/pull/961. M. H. M. Winands, Y. Björnsson and J.-T. Saito,
"Monte-Carlo tree search solver", Computers and Games 2008. The Copenhagen rules:
https://aagenielsen.dk/copenhagen_rules.php.

Copyright © 2026 Devin O'Keefe. This program is free software under the
[GNU General Public License](LICENSE), version 3 or (at your option) any later version, and comes
with no warranty. It is TaflZero's licence: to match TaflZero exactly, the scan for shieldwall
captures in `tafl-core`, and its test of which defenders of an exit fort could still be captured,
follow TaflZero's code step by step. To cite the project, see [CITATION.cff](CITATION.cff).
