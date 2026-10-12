# Results

Matches against TaflZero (the build described in the README, with the g0213 network) under
`taflzero-copenhagen-operational-v1`. The target is to beat it at 15 seconds per move
for both engines (the Fetlar championship limit), one game at a time on this machine
with training paused; shorter matches at 400 nodes per move track progress. Each
pair plays one random two-ply opening twice with the sides swapped; the confidence
interval and sign test use the pair as the unit. "Nodes" is the search budget per
move for both engines: new playouts, on top of the tree kept from earlier moves
(TaflZero always keeps its tree; ours did not before 2026-10-03 22:30).

Matches played while training runs understate TaflZero, which suffers far more from
a busy GPU than our search does (about 280 instead of 600
nodes per second at 1 s per move). Only matches marked "training paused" are fair
comparisons; the others track trends.

Other jobs on this shared machine have the same effect, so a timed match can apply
a speed rule (`--min-opponent-speed`): a game counts only if TaflZero averaged that
many nodes per second over its moves. Slower games stay in the record and are
played again, whatever their result. The limit is 80% of the speed TaflZero reaches
on the idle machine. For the 15 s match that was first taken as 1,200 nodes per
second (idle about 1,500); replayed games run above normal priority, where idle is
about 1,700, and since 2026-10-05 02:00 the limit is 1,350. "Idle" proved to depend
on the processor jobs of the day: on the evening of 2026-10-05, with fewer of them,
the same test reached 2,100 to 2,270. The limit was left where it had been set, at
about 60% of that figure, and the match's row gives the result under stricter limits
too. From 2026-10-05 the
arena also watches the GPU: a game during which another process was using it is
stopped, or marked if it had just ended, and is played again as well. So is a game
in which five TaflZero searches in a row averaged less than three quarters of the
limit: one replayed win had such a stretch (about 500 nodes per second for two
minutes) that the GPU watch missed.

Node counts in the notes and records up to 2026-10-10 are not comparable between the
engines: ours is the root's visit count after the search, which includes the part of
the tree kept from earlier moves, and TaflZero's is what it reports for the search.
From 2026-10-11 our count is the simulations of the search itself.

| Date | Network | Budget | Pairs | Games W-L | Score | Notes |
|---|---|---|---|---|---|---|
| 2026-10-03 | r1 step 1,699 (~55 min of training, 437k samples) | 400 nodes | 16 | 0-32 | 0.00 (16 pairs lost, sign test p = 3e-5) | Baseline. TaflZero won every game: as defender by escape in about 26 plies (14 games) or by an exit fort (2), as attacker mostly by leaving our defenders without a move. Our engine used 4.8 s per move against TaflZero's 0.5 s, before CUDA-graph inference. |
| 2026-10-03 | r1 step 6,000 (1.5M samples) | 400 nodes | 16 | 0-32 | 0.00 (p = 3e-5) | Games much longer (129 plies on average). As defender TaflZero won every game, 13 by escape and 3 by exit fort; as attacker it won all 16 by a third repetition, which led to the repetition tactic. No tree reuse on our side. |
| 2026-10-03 to 10-04 | r1 step 10,000 (2.6M samples, ~4 h of training) | 400 nodes | 16 | 0-16, 16 unfinished | 0.25 (16 pairs lost, p = 3e-5) | With the repetition tactic and tree reuse. Our defence held all 16 of TaflZero's attacks to the 600-ply limit; our attack lost all 16 (TaflZero escaped or built an exit fort in 36-96 plies). Mean game 332 plies. |
| 2026-10-04 | r1 step 16,000 (4.1M samples, ~6 h) | 400 nodes | 16 | 3-18, 11 unfinished | 0.27, Elo -177 [-274, -100] (1 pair won, 13 lost, p = 0.002) | First wins: one as attacker (king captured), two as defender (escapes). As defender: 11 held to 600 plies, 3 lost to repetition. As attacker: 15 of 16 lost (TaflZero built exit forts in 10). |
| 2026-10-04 | r1 step 18,000 (4.6M samples, ~7 h) | 1 s per move | 16 | 2-16, 14 unfinished | 0.28, Elo -163 [-201, -128] (14 pairs lost, p = 1e-4) | First timed match, 16 leaves per batch, training running alongside. Our engine searched about 1,100 nodes per move to TaflZero's 450. As defender: 14 held to 600 plies, 1 escape, 1 lost to repetition. As attacker: 1 win (defenders left without a move), 15 losses, 13 of them to exit forts built in 30-66 plies. |
| 2026-10-04 | r1 step 26,000 (6.7M samples, ~9.5 h) | 1 s per move | 16 | 20-4, 8 unfinished | 0.75, Elo +191 [+100, +315] (12 pairs won, 1 lost, p = 0.003) | First match with the exit-fort tactic (from step 22,137). As attacker: 12 wins (4 king captures, 8 by leaving the defenders without a move), 4 losses to exit forts. As defender: 8 escapes, 8 held to 600 plies. Caveat: training ran alongside and TaflZero searched only ~280 nodes per move (450 at step 18,000; ours ~1,240). The paused rerun below is the valid measurement. |
| 2026-10-04 | r1 step 26,000, training paused | 1 s per move | 16 | 11-11, 10 unfinished | 0.50, Elo 0 [-96, +96] (5 pairs won, 7 lost, p = 0.77) | Same match without contention: TaflZero ~600 nodes per move, ours ~1,300. As attacker: 6 wins (2 king captures, 2 repetitions, 1 encirclement, 1 no moves), 10 losses, 9 of them exit forts. As defender: 5 escapes, 10 held to 600 plies, 1 lost to repetition. |
| 2026-10-04 | r1 step 18,000, training paused (control) | 1 s per move | 6 of 16 | 1-17 | (match stopped after 18 games) | The step-18,000 match replayed without contention, with its original engine library. As attacker: 1 win, 11 exit-fort losses; as defender: all 6 lost to repetition. The arena stopped when TaflZero stopped answering, at a time when other jobs had exhausted the machine's memory. |
| 2026-10-04 | r1 step 36,000 (9.2M samples, ~13 h), training paused | 1 s per move | 16 | 20-3, 9 unfinished | 0.77, Elo +206 [+103, +359] (13 pairs won, 2 lost, p = 0.007) | TaflZero ~590 nodes per move, ours ~1,200. As attacker: 13 wins (5 king captures, 4 repetitions, 2 encirclements, 2 no moves), 3 losses (2 escapes, 1 no moves), no exit forts (9 at step 26,000). As defender: 7 wins (4 escapes, 3 exit forts of our own), 9 held to 600 plies, no losses. Forts had risen from ~1% to ~9% of self-play games after the fort tactic and fort-threat exploration (step 22,710). |
| 2026-10-04 | r1 step 36,000, training paused | 15 s per move, one game at a time | 8 | 14-0, 2 unfinished; under the speed rule 3-0, 1 unfinished (4 games) | 0.94 over all 16 games (8 pairs won, p = 0.008), but only 4 games count | First match at the target time (16 leaves per batch). As attacker: 8 wins (3 king captures, 2 encirclements, 2 no moves, 1 repetition). As defender: 6 exit forts, 2 held to 600 plies. Caveat: from the fifth game on, other jobs filled the processor and later the GPU, and slowed both engines. TaflZero averaged 1,450-1,540 nodes per second in games 1-4 and 1,081-1,222 in games 9-10, but 360-540 in games 5-8 and 11, and 40-100 in games 12-16. Games 1-4 pass the speed rule (3 wins, 1 unfinished; both pairs won). The other 12 did not count and were played again: 11 of them had been wins, 1 unfinished. The next row has the match as it counts. |
| 2026-10-04 to 10-06 | r1 step 36,000, training paused | 15 s per move, one game at a time, speed rule (1,350 nodes per second) | 8 | 12-1, 3 unfinished | 0.84, Elo +293 [+160, +620] (7 pairs won, none lost, 1 tied, p = 0.016) | The target match as it counts: games 1-4 of the first run and 12 games replayed between 2026-10-05 01:00 and 2026-10-06 07:46, at above normal priority, at times when no other process was on the GPU. As attacker: 7 wins (3 by leaving the defenders without a move, 2 encirclements, 2 repetitions) and 1 loss (the king escaped after 140 plies). As defender: 5 wins (3 exit forts, 2 escapes) and 3 games held to 600 plies. TaflZero averaged 1,373 to 2,167 nodes per second in these games (median 1,660) and used 13.8 to 15.0 s per move; our engine used 12.7 to 15.0 s. Under a stricter limit the result holds on fewer games: of the 12 in which TaflZero averaged 1,500 or more, 10 were won, 1 lost and 1 unfinished; all 6 at 1,680 or more were won. Caveats: 16 games are a small sample. The machine was never idle: before a game TaflZero reached between 1,390 and 2,270 nodes per second on the test position, depending on how many processor jobs were running. Our engine slowed with it, roughly in step: comparing the busier half of the replay games with the quieter half, the first search of a game was 20-25% slower for TaflZero and 13-23% slower for ours. Games 1-4 were recorded before single searches were logged, so the dip rule could not be applied to them. Besides the 12 slow games of the first run, 7 replay games do not count: 6 were stopped when another process took the GPU or TaflZero's speed dipped, and 1 win was set aside by the dip rule. |
| 2026-10-11 to 10-12 | r1 step 468,000 (119.8M samples, 7.7 days), training paused | 15 s per move, one game at a time, speed rule (1,350 nodes per second) | 8 | 14-0, 2 unfinished | 0.94, Elo +470 [+311 or more] (8 pairs won, none lost, p = 0.008) | Same openings, rules and settings as the step-36,000 match; this network beat that one 32-0 at 800 nodes. As attacker: 8 wins (6 king captures, 1 encirclement after 413 plies, 1 by leaving the defenders without a move); apart from the encirclement they took 83 to 153 plies. As defender: 6 exit forts in 36-62 plies, 2 games held to 600 plies. TaflZero averaged 1,650 to 3,066 nodes per second (median 2,670), faster than in the step-36,000 match because the machine was quieter: on the test position before a game it reached 1,770 to 3,210 nodes per second, against 1,390 to 2,270 before the replays of that match. It used 13.9 to 15.0 s per move; our engine used 11.5 to 15.0 s. Under a stricter limit the result holds too: in the 12 games in which TaflZero averaged 2,400 or more, our engine won 11, lost none and left 1 unfinished. Our engine ran 6,400 to 33,800 simulations per second (median 14,900); in this match these are the search's own simulations, comparable with TaflZero's figures. A restart of the computer at 2026-10-12 00:09 ended the match process during the first game of the seventh pair; that game left no record. Five games were stopped when another process took the GPU, and played again: one before the restart and four after it, all of them the first game of the seventh pair. The interval is a normal approximation over the 8 pair scores, a rough one here; its upper end is above a score of 1, which has no Elo equivalent, so only the lower end is given. |

## Between our own checkpoints

Training went on after the step-36,000 match at 15 s, to step 471,564 on 2026-10-11
(120.7M samples, 2.6M self-play games). These matches measure what it added: 800 nodes per
move for both sides (8 leaves per batch), 16 pairs, training running alongside,
which is harmless here because both sides are our engine.

| Date | Match | Games W-L | Score | Notes |
|---|---|---|---|---|
| 2026-10-11 | step 468,000 against step 36,000 | 32-0 | 1.00 (16 pairs won, p = 3e-5) | The network of step 36,000 lost every game, as attacker and as defender; games took 119 plies on average. |
| 2026-10-11 | step 148,000 against step 36,000 | 28-4 | 0.88, Elo +338 (12 pairs won, none lost, p = 5e-4) | |
| 2026-10-11 | step 300,000 against step 148,000 | 19-13 | 0.59, Elo +66 (5 pairs won, 2 lost, p = 0.45) | Not significant. |
| 2026-10-11 | step 468,000 against step 300,000 | 14-15, 3 unfinished | 0.48 (2 pairs won, 2 lost) | No progress in 168,000 steps. Between these two the attackers won 21 of the 29 decided games. |

Progress had stopped at the constant learning rate of 0.02, as the training loss
also showed (2.32 at step 300,000, 2.35 at step 469,500). From step 471,564 the
learning rate is 0.005.

The records of every match above are in `matches/` ([matches/README.md](../matches/README.md)), and
`python tools/check_results.py --check` recomputes this file's results from them.
