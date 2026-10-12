# Training records

The records of the training run `r1`, from which every network in
[docs/RESULTS.md](../docs/RESULTS.md) comes, written by `python tools/training_records.py RUN_DIR`
from the run's folder. [tools/check_results.py](../tools/check_results.py) takes the training
steps, samples, games and times that the results quote from these files. The two checkpoints that
the results use are not distributed; [models/manifest.json](../models/manifest.json) gives their
sizes, SHA-256 and counters.

## r1-config.json

The run's configuration (`TrainConfig` in [python/tafl/train.py](../python/tafl/train.py)) as it
is now. Training was restarted with it at step 471,564 with a lower learning rate: `lr` is 0.005
from then on. Before, after the warm-up of 500 steps, it was 0.02, as the `lr` column of the
metrics shows.

## r1-metrics.csv

One row for each step at which the trainer logged its metrics (every 100 steps). A restart after a
crash goes on from the last saved state, so a few steps were logged twice (27,100, 29,600, 36,800
and 36,900); the row logged last is kept.

| Column | Meaning |
|---|---|
| `time` | When the row was logged, in seconds since 1970-01-01 UTC. |
| `step` | Training steps (batches of 1,024 samples) taken. |
| `generated`, `games` | Samples (positions with their search results) and games that self-play had generated in all. |
| `lr` | The learning rate of the step. |
| `mean_plies`, `attacker_wins`, `undecided` | Over the trainer's last 2,000 self-play games, or fewer: the mean length in plies, the share of the decided games that the attackers won, and the share of games without a winner. The trainer keeps these games in memory, so a restart empties the window and the first rows after one cover fewer games. |
| `loss` | The training loss, the weighted sum of the terms below (weights in the configuration), averaged over the steps since the last row. |
| `policy`, `policy_kl` | Cross-entropy of the policy against the search's visit distribution, and the same less the distribution's entropy. |
| `opponent` | Cross-entropy of the opponent-policy head against the move the opponent played next. |
| `value`, `q`, `moves_left` | The value head's loss against the game's result, the Q head's against the search's root value, and the moves-left head's. |

The values are rounded (losses to four decimals), with trailing zeros dropped.

## r1-events.txt

The lines of the trainer's log that are not a logged step: starts (with the samples generated and
loaded), stops, crashes and the supervisor's restarts. Times are the machine's local time, UK summer
time (+01:00).

## r1-selfplay.csv

Every self-play game of the run, from the 3.1 GB `selfplay/games.jsonl`, in blocks of 2,000 in the
order they were played: the block's first game (from 1), its games, the attackers' and defenders'
wins and the unfinished games, the mean length in plies, how the games ended (columns named as in
the match records, [matches/README.md](../matches/README.md)), and `approx_step`. The file was read
once, from start to end; no line was torn by a crash. Self-play games are capped at 400 plies
(`max_plies`); a game unfinished at the cap counts in `unfinished`.

`approx_step` is the first logged step at which the trainer's game counter had reached the end of
the block: about the step at which the block's games were played. It is approximate. After a crash the
trainer goes on from the counters it last saved, while the games played since stay in the file, so
the file runs ahead of the counter (by 6,064 games at the end of the run), and the counter goes back
a little; it is taken as never decreasing. The last blocks, past the last count, have none.

## Not copied

The run's folder also holds what this repository leaves out. Up to the checkpoint of step 468,000
its `selfplay/` folder had 63,594 files of training samples, 60.7 GB, beside the 3.1 GB
`games.jsonl`; the checkpoints, one every 2,000 steps and one at each stop, are 12.8 MB each, and none is
distributed.
