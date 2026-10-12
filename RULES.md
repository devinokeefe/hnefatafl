# Copenhagen Hnefatafl, 11×11, as this engine plays it

This is the exact ruleset that the engine, the training and the match referee implement: the
Copenhagen rules as TaflZero implements them, in the build that the matches were played against
(see "The opponent" in the README). The code and docs/RESULTS.md call it
`taflzero-copenhagen-operational-v1`. Following that build lets the two engines replay each other's
moves in a match and never disagree about a position.

Where the official rules leave a case open, or Copenhagen implementations differ, this ruleset does
what that build of TaflZero does; those cases are listed under "Interpretation notes". One rule
here, that a side with no legal move loses, is in the official rules and in that build, but not in
the published TaflZero, which does not check for it.

The code is in `crates/tafl-core` (`position.rs` for moves and captures, `terminal.rs` for the end of
the game, `game.rs` for repetition). `tafl difftest` compares it with TaflZero's rules position by
position (see the README).

## Board

An 11×11 board of squares. Files are `a`–`k` from left to right and ranks `1`–`11` from bottom to top,
so the corners are `a1`, `k1`, `a11` and `k11`, and the centre square, the **throne**, is `f6`.
The four corners and the throne are the **restricted squares**.

## Pieces and setup

* 24 **attackers**, in four groups of six around the middle of each edge: `d1`–`h1` and `f2`, and the
  same on the other three edges.
* 12 **defenders** and the **king**: the king on the throne, the defenders in a diamond around it
  (`f4`, `e5`–`g5`, `d6`, `e6`, `g6`, `h6`, `e7`–`g7`, `f8`).
* The attackers move first, and the sides then alternate.

In the notation used throughout (TaflZero's), the start position is

```
3aaaaa3/5a5/11/a4d4a/a3ddd3a/aa1ddkdd1aa/a3ddd3a/a4d4a/11/5a5/3aaaaa3 a
```

rank 11 first, `a` an attacker, `d` a defender, `k` the king, a digit a run of empty squares, and the
last letter the side to move.

## Moves

1. Every piece moves like a rook in chess: any number of empty squares along a rank or a file. It may
   not jump over a piece.
2. Only the king may stop on a restricted square. Any piece may pass over the throne when it is empty.
3. A move is written as its two squares, for example `d1d4` or `f11f8`.

## Captures

After a move, captures are resolved in this order. Only the side that moved captures.

1. **Sandwich.** An enemy piece next to the square the moving piece arrived on is captured if the
   square beyond it, on the same line, is *hostile* to it. Hostile squares are those holding a piece
   of the moving side (the king counts as a defender, so the king captures) and the empty restricted
   squares (an empty corner, or the empty throne, is hostile to both sides). A piece may move between
   two enemies without being captured. The king is never captured by a sandwich.
2. **Shieldwall.** If the moving piece arrived on an edge, that edge is scanned from one corner to
   the other. A run of two or more enemy pieces along the edge, each with a piece of the moving side
   on the square in front of it (towards the middle of the board), and bracketed at both ends by
   pieces of the moving side or by corners, is captured all at once, provided the moving piece is one
   of the two brackets. The king can be part of such a run but is never removed by it; the rest of
   the run is. (See "Interpretation notes" for a run of a single piece.)
3. **The king.** After an attacker's move next to the king, the king is captured if each of its four
   neighbours is an attacker or the throne. So the king cannot be captured on an edge, and next to
   the throne three attackers are enough.

## End of the game

After every move the following are checked in this order, and the first that applies ends the game.

1. The king has been captured: **the attackers win**.
2. The position (the board and the side to move) occurs for the **third time**: **the attackers
   win**. The start position is not counted.
3. The king stands on a corner: **the defenders win**.
4. **Encirclement:** no defender, the king included, can reach any edge square by moving through
   squares not held by attackers. **The attackers win.** This is checked only after an attacker's
   move that captured something or that arrived next to at least two other attackers (counting the
   eight surrounding squares).
5. **Exit fort:** **the defenders win** if the king stands on an edge, has at least one empty
   neighbouring square, and no attacker can ever reach it. Precisely: the region of squares the king
   can reach through squares not held by defenders contains no attacker, and stays so after removing,
   round by round, every defender that the attackers could conceivably capture. An *attacker area*
   is a set of orthogonally connected squares, each holding an attacker or empty, grown from an
   attacker; restricted squares belong to no area. A defender counts as capturable when the two
   squares beside it on one line both lie in attacker areas, at least one of the two areas has an
   empty square, and, if both squares are in the same area, that area holds at least two
   attackers. This is checked only after a king move, or after a defender move that arrived
   next to at least two defenders or edge squares, counting the eight surrounding squares but not the
   king (one is enough if the defender itself arrived on an edge).
6. The side to move has no legal move: **it loses**.

## Interpretation notes

These are the cases where the official text is open or Copenhagen implementations differ.

* **A third occurrence of a position wins for the attackers**, whoever repeated it. The official
  text says only that perpetual repetition is forbidden and loses for the defenders; the third
  occurrence is TaflZero's reading of it. The start position is not counted, so the first return to
  it is its first occurrence.
* **Encirclement and exit forts are tested only after the moves listed above.** In TaflZero this
  gating is part of the rules, not only a shortcut: a fort or an encirclement that arises from some
  other move is not recognised until a qualifying move is played.
* **The exit fort is TaflZero's test**, which removes every defender the attackers could conceivably
  capture, not only those they could capture at once.
* **Shieldwall scan.** TaflZero's scan keeps a lone enemy piece that stands right after a bracket in
  the run that follows, instead of starting afresh, so such a piece can be captured together with a
  later run. The engine reproduces this; `shieldwall_single_piece_quirk_matches_taflzero` in
  `crates/tafl-core/src/tests.rs` is an example.
* **The king is armed**: it takes part in sandwiches as a hammer and as an anvil.
* **The empty throne is hostile to both sides**, and a throne occupied by the king is hostile only to
  attackers (as a defender piece).

## Matches and self-play

These are not rules of the game but limits applied by the programs. A match game still running after
600 plies is stopped and counted as unfinished (half a point to each side in the score). Self-play
games stop at 400 plies and are recorded without a winner.
