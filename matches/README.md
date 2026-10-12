# Match records

The records of every match that [docs/RESULTS.md](../docs/RESULTS.md) reports: each game as the
arena (`python -m tafl.arena`) recorded it, the arena's log of each run, and its summary.
[tools/check_results.py](../tools/check_results.py) recomputes every number of the results from
these files; `python tools/check_results.py --json PATH` writes every game with whether it counts
and why not.

## Files

A match is named after the network and the search budget: `s36000-t15000` is the checkpoint of
training step 36,000 at 15,000 ms per move, `s1699-n400` the one of step 1,699 at 400 nodes per
move. `-paused` means training was stopped while the match was played; `ladder-s468000-vs-s36000-n800`
is a match between two of our checkpoints at 800 nodes per move.

| File | What it is |
|---|---|
| `<match>.jsonl` | The game records, one game per line in the order they were played, copied byte for byte (with the arena's Windows line endings, which `.gitattributes` keeps). |
| `<match>.json` | The header of each run, when its log was created and last written, the SHA-256 and number of the game records, and the arena's summary. |
| `<match>.log`, `<match>.2.log`, ... | The arena's output for each run. A match continued with `--resume` has one log per run. |
| `<match>.err.log` | The error output of a run that wrote any (only `s18000-t1000-paused`, which stopped when TaflZero did). |
| `paused_controls.txt` | The log of the script that stopped training, played `s26000-t1000-paused` and `s18000-t1000-paused`, and restarted training. |

They were copied from the arena's folder by `python tools/copy_records.py RUNS_DIR`. The copy
refuses to change or drop a game line that an earlier copy holds, and leaves out a last line still
being written, so copying a match in progress can only add games. In the logs and headers, absolute
paths are replaced: a checkpoint by its file name (`models/<name>` for the two in
[models/manifest.json](../models/manifest.json), which are not distributed), a file of the checkout by its path in the repository, and a file of the
Python installation by `<python>/<path>`. The times are the file times on the machine that played
the matches, in its time zone (UK summer time, +01:00).

## The game records

Each line of a `.jsonl` file is one game:

| Key | Meaning |
|---|---|
| `pair` | The opening pair, from 0. Each opening is played twice, each engine attacking once. |
| `opening` | The two random plies that start both games of the pair (from the header's `seed`), space-separated, e.g. `a4c4 e5e2`. |
| `attackers`, `defenders` | Who played each side: `ours`, `taflzero`, or in a ladder match `opponent` (the older checkpoint). |
| `moves` | Every ply of the game, the opening included, space-separated. A move is the square moved from and the square moved to (files `a` to `k`, ranks 1 to 11). |
| `busy` | Present, and `true`, only if the arena stopped or marked the game because another process was using the GPU, or (from 2026-10-05) because TaflZero's speed dipped while it was played. |
| `winner` | `attackers`, `defenders`, or `null` for a game that did not finish. |
| `reason` | How it ended: `KingCaptured`, `Encircled` or `Repetition` (a third repetition, which the attackers win) for the attackers; `KingEscaped` or `ExitFort` for the defenders; `NoMoves` (the side to move had no legal move, and lost); `PlyCap` (unfinished at the 600-ply limit) or `Interrupted` (stopped as busy) for a game without a winner. |
| `plies` | The length of the game in plies, the opening included. |
| `<player>_mean_nodes` | The mean nodes per move that the player's search reported, over its moves of the game. Up to 2026-10-10 the two engines counted different things (docs/RESULTS.md); from 2026-10-11 ours counts the simulations of its search. |
| `<player>_mean_ms` | The mean time per move of the player's search, in milliseconds (wall time, as the arena measured it). |
| `<player>_searches` | Each of the player's searches in order, as `[nodes, ms]` (ms rounded to whole milliseconds). Recorded from the second run of `s36000-t15000-paused` on: its first 16 games and the matches before it have none. |

A player's speed in a game, used below, is `1000 * mean_nodes / mean_ms` nodes per second.

The header (the first line of each log, and `runs[].header` in `<match>.json`) has the arena's
settings: `model` and `opponent` (checkpoints, or `taflzero`), `ours` and `theirs` (each side's
budget: `400 nodes` or `15000 ms`), `search` (our search's algorithm and its constants), `leaves`
(leaves evaluated per batch), `reuse_tree`, `moves_left_alpha`, `pairs`, `opening_plies`, `seed`,
and in later runs `min_opponent_speed` (the speed rule's limit, below; 0 for none),
`above_normal` (whether the engines ran above normal process priority) and `gpu_priority`
(whether they ran in the realtime class of the Windows GPU scheduler). Keys that a run's arena did
not yet have are absent. `--attempts` (below) is not recorded; the default is 3, and with these
records any value from 2 up counts the same games.

## Which games count

Every game of a match without a speed rule counts. The 15 s matches have one: the header's
`min_opponent_speed` is a speed, in TaflZero's nodes per second, that TaflZero must reach for a
game to say something about TaflZero at full strength. A game counts only if

- TaflZero's speed in the game is at least the limit;
- it is not `busy`: from 2026-10-05 the arena watches the GPU and stops a game when another
  process uses it, or marks one that had just ended;
- no five consecutive searches of TaflZero that took a second or more averaged under three
  quarters of the limit (the dip rule, from 2026-10-05; the arena also stops a game in which this
  happens). The single searches are needed for this, so games recorded without them are not
  checked.

A game that does not count is played again: one that was busy or dipped always, one that was
merely slow until it had been slow `--attempts` times. When a match is resumed (`--resume`), the
arena reads its records in order with the rule of the new run: for each scheduled game (each pair
and attacking side) the first record that the rule lets count is counted, and the records after it
of the same game are passed over. So the limit of a match's last run decides for all its records.
`s36000-t15000-paused` was first played without a limit, resumed with a limit of 1,200 and then
1,350; its games count at 1,350. Its first run's own summary, with all 16 games, is
`first_run_summary` in its `.json`.

## The summary

`summary` in `<match>.json` is what the arena printed at the end of the match's last run, without
the header that it repeats: the counted games' `games`, `wins`, `losses` and `unknown`
(unfinished) for our engine; `as_attackers` and `as_defenders`, our mean score on each side (a win
1, an unfinished game 0.5); and over the complete pairs (both games counted) `pairs`, `score` (the
mean pair score) with `score_ci95` (mean ± 1.96 standard errors, the variance with n - 1), `elo`
and `elo_ci95` (the Elo difference of those scores, clamped at ±2,400 for a score of 0 or 1), `z`,
`pairs_won` and `pairs_lost` (pairs scoring over and under 0.5), `sign_test_p` (the two-sided sign
test over the pairs won and lost) and `not_counted` (the records set aside). It is `null` for a
match whose last run did not finish.
