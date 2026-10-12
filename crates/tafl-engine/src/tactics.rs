//! Exact short tactics used when a search node is expanded.
//!
//! Hnefatafl is decided by king escapes, king captures and (for the attackers) third
//! repetitions, and a network easily misjudges a single forcing move. Before asking
//! the network, the search therefore checks (exactly, under the full rules) whether
//! the side to move wins at once, and whether it faces an immediate threat; under a
//! threat only the moves that parry it (or win outright) are kept, since every other
//! move loses on the spot.

use tafl_core::geometry::{CORNERS, DIRECTIONS, EDGES, SIZE, THRONE_BB, col_of, is_edge, row_of, square};
use tafl_core::{Bb, Move, MoveList, Position, REPETITION_LIMIT, Reason, Side, Sq};

use crate::encode::king_can_escape;

// Built once per node expansion and consumed at once: boxing the move list would only
// add an allocation.
#[allow(clippy::large_enum_variant)]
pub enum Analysis {
    /// The side to move wins immediately with this move.
    Win(Move),
    /// Every legal move loses immediately (or there is none).
    Loss,
    /// Candidate moves: all legal moves, or only those that meet a threat.
    Moves(MoveList),
}

/// The square from which an attacker move would capture the king, if one exists.
pub fn king_capture_square(pos: &Position) -> Option<Sq> {
    if !pos.has_king() || is_edge(pos.king) {
        return None;
    }
    let around = Bb::from_sq(pos.king).neighbors();
    let open = around & !(pos.attackers | THRONE_BB);
    if open.count() != 1 {
        return None;
    }
    let gap = open.first();
    if pos.occupied().has(gap) {
        return None;
    }
    // Some attacker must be able to slide onto the gap (it is never a restricted square:
    // a non-edge king has no corner neighbours, and the throne counts as hostile).
    let occupied = pos.occupied();
    let (r, c) = (row_of(gap) as isize, col_of(gap) as isize);
    for (dr, dc) in DIRECTIONS {
        let (mut rr, mut cc) = (r + dr, c + dc);
        while rr >= 0 && rr < SIZE as isize && cc >= 0 && cc < SIZE as isize {
            let sq = square(rr as usize, cc as usize);
            if occupied.has(sq) {
                if pos.attackers.has(sq) {
                    return Some(gap);
                }
                break;
            }
            rr += dr;
            cc += dc;
        }
    }
    None
}

/// A move with which the attackers capture the king right now.
fn king_capture_move(pos: &Position) -> Option<Move> {
    let gap = king_capture_square(pos)?;
    pos.attackers.iter().find(|&from| pos.destinations(from).has(gap)).map(|from| Move::new(from, gap))
}

/// An attacker move that brings about a position for the third time, winning at
/// once. Only a move without captures can repeat a position (a capture leaves less
/// material than any earlier position had), so quiet hashes find the candidates.
fn repetition_win(pos: &Position, moves: &MoveList, occurrences: &impl Fn(u64) -> u32) -> Option<Move> {
    moves.iter().copied().find(|&mv| {
        if occurrences(pos.quiet_hash_after(mv)) + 1 < REPETITION_LIMIT {
            return false;
        }
        let mut next = *pos;
        next.play(mv);
        next.outcome(occurrences(next.hash) + 1).is_some_and(|o| o.winner == Side::Attackers)
    })
}

/// A move of a defender other than the king that completes an exit fort (defenders to
/// move); forts the king completes are found among the decisive king moves.
///
/// The fort test needs the king on the edge. While attackers can reach the king's
/// area (the squares it reaches through non-defenders), a move can only shut them out
/// by landing in that area: a capture beside its destination, even one starting a
/// shieldwall, would otherwise put the destination in the area too. And unless the
/// destination is on the edge (where a shieldwall can capture further away), the
/// attackers left in the king's part of the area must all stand beside it. Every
/// other move is skipped without being played.
fn fort_move(pos: &Position, occurrences: &impl Fn(u64) -> u32) -> Option<Move> {
    if !pos.has_king() || !is_edge(pos.king) {
        return None;
    }
    let king = Bb::from_sq(pos.king);
    let area = Bb::flood(king, !pos.defenders);
    let open = (area & pos.attackers).any();
    let shuts_out = |to: Sq| {
        let to_bb = Bb::from_sq(to);
        is_edge(to) || (Bb::flood(king, area & !to_bb) & pos.attackers & !to_bb.neighbors()).is_empty()
    };
    for from in pos.defenders {
        let targets = if open { pos.destinations(from) & area } else { pos.destinations(from) };
        for to in targets {
            if open && !shuts_out(to) {
                continue;
            }
            let mv = Move::new(from, to);
            let mut next = *pos;
            next.play(mv);
            if next.outcome(occurrences(next.hash) + 1).is_some_and(|o| o.reason == Reason::ExitFort) {
                return Some(mv);
            }
        }
    }
    None
}

/// Whether a defender move (defenders to move in `pos`) threatens to complete an exit
/// fort with the defenders' next move, by the king stepping to the edge or another
/// defender closing the king's area. A move that ends the game is not a threat.
pub fn threatens_fort(pos: &Position, mv: Move, occurrences: &impl Fn(u64) -> u32) -> bool {
    debug_assert!(pos.side == Side::Defenders);
    let mut next = *pos;
    next.play(mv);
    if next.outcome(occurrences(next.hash) + 1).is_some() {
        return false;
    }
    next.set_side(Side::Defenders);
    let king = next.king;
    let fort = |to: Sq| {
        let mut after = next;
        after.play(Move::new(king, to));
        after.outcome(occurrences(after.hash) + 1).is_some_and(|o| o.reason == Reason::ExitFort)
    };
    (next.destinations(king) & EDGES & !CORNERS).iter().any(fort) || fort_move(&next, occurrences).is_some()
}

/// Analyses a non-terminal position. `occurrences(hash)` must give how often a
/// position has occurred so far (so that repetition endings are judged exactly).
///
/// Besides one-move wins (escapes, king captures, third repetitions, exit forts) and
/// threats, this finds escapes in two king moves: a king
/// move after which no attacker move stops the escape is a proven win for the
/// defenders, and the attackers keep only moves that leave no such king move.
pub fn analyse(pos: &Position, occurrences: &impl Fn(u64) -> u32) -> Analysis {
    let moves = pos.legal_moves();
    match pos.side {
        Side::Defenders => {
            if king_can_escape(pos) {
                let to = (pos.destinations(pos.king) & CORNERS).first();
                return Analysis::Win(Move::new(pos.king, to));
            }
            if let Some(mv) = decisive_king_move(pos, occurrences, &[]).or_else(|| fort_move(pos, occurrences)) {
                return Analysis::Win(mv);
            }
            let mut threatened = *pos;
            threatened.set_side(Side::Attackers);
            if king_capture_square(&threatened).is_none() {
                return Analysis::Moves(moves);
            }
            parry(pos, &moves, occurrences, |next| king_capture_square(next).is_some())
        }
        Side::Attackers => {
            if let Some(mv) = king_capture_move(pos).or_else(|| repetition_win(pos, &moves, occurrences)) {
                return Analysis::Win(mv);
            }
            let mut threatened = *pos;
            threatened.set_side(Side::Defenders);
            // An immediate escape threat also shows up as a decisive king move.
            let decisive = decisive_king_move(&threatened, occurrences, &[]);
            if decisive.is_none() && fort_move(&threatened, occurrences).is_none() {
                return Analysis::Moves(moves);
            }
            parry(pos, &moves, occurrences, |next| {
                decisive_king_move(next, occurrences, decisive.as_slice()).is_some()
                    || fort_move(next, occurrences).is_some()
            })
        }
    }
}

/// Whether the attackers, to move with the king on the edge and a corner in its
/// reach, have a move that stops the escape or wins at once.
///
/// A king on the edge cannot be captured (that takes four sides) or encircled, and
/// keeps a move while its way to the corner is open, so an attacker move can only
/// stop it by landing on a square the king can reach, or by repeating a position for
/// the third time. Only a move without captures can repeat one (a capture leaves less
/// material than any earlier position had), so its quiet hash tells. Every other move
/// is skipped without being played.
fn attackers_can_hold(pos: &Position, occurrences: &impl Fn(u64) -> u32) -> bool {
    debug_assert!(pos.side == Side::Attackers && is_edge(pos.king));
    let reach = pos.destinations(pos.king);
    pos.legal_moves().iter().any(|&mv| {
        if !reach.has(mv.to) && occurrences(pos.quiet_hash_after(mv)) + 1 < REPETITION_LIMIT {
            return false;
        }
        let mut next = *pos;
        next.play(mv);
        match next.outcome(occurrences(next.hash) + 1) {
            Some(o) => o.winner == Side::Attackers,
            None => !king_can_escape(&next),
        }
    })
}

/// A king move (defenders to move in `pos`) that escapes, or after which the
/// attackers cannot stop an escape. The moves in `first` are tried first.
///
/// Only moves to the edge are tried: a corner can only be reached along an edge, and
/// the other wins a king move could bring (the attackers left without a move) are
/// left to the search, which keeps the analysis sound if not complete.
fn decisive_king_move(pos: &Position, occurrences: &impl Fn(u64) -> u32, first: &[Move]) -> Option<Move> {
    if !pos.has_king() {
        return None;
    }
    let king = pos.king;
    let reach = pos.destinations(king) & EDGES;
    let hinted = first.iter().copied().filter(|m| m.from == king && reach.has(m.to));
    let rest = reach.iter().map(|to| Move::new(king, to)).filter(|m| !first.contains(m));
    hinted.chain(rest).find(|&mv| {
        let mut next = *pos;
        next.play(mv);
        match next.outcome(occurrences(next.hash) + 1) {
            Some(o) => o.winner == Side::Defenders,
            None => king_can_escape(&next) && !attackers_can_hold(&next, occurrences),
        }
    })
}

/// Keeps the moves after which `still_lost` is false, returning a win at once if
/// some move ends the game in the mover's favour.
fn parry(
    pos: &Position,
    moves: &MoveList,
    occurrences: &impl Fn(u64) -> u32,
    still_lost: impl Fn(&Position) -> bool,
) -> Analysis {
    let mut kept = MoveList::new();
    let mut any = false;
    for &mv in moves.iter() {
        let mut next = *pos;
        next.play(mv);
        match next.outcome(occurrences(next.hash) + 1) {
            Some(o) if o.winner == pos.side => return Analysis::Win(mv),
            Some(_) => continue,
            None => {}
        }
        if !still_lost(&next) {
            kept.push(mv);
            any = true;
        }
    }
    if any { Analysis::Moves(kept) } else { Analysis::Loss }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tafl_core::{Game, parse_move};

    fn analyse_game(game: &Game) -> Analysis {
        analyse(game.position(), &|h| game.occurrences(h))
    }

    #[test]
    fn finds_escape_and_capture() {
        let game = Game::from_fen_and_moves("11/11/11/11/11/11/11/11/11/a10/1k7a1 d", &[]).unwrap();
        assert!(matches!(analyse_game(&game), Analysis::Win(m) if m == parse_move("b1a1").unwrap()));

        let game = Game::from_fen_and_moves("11/11/2a8/1aka7/11/11/11/11/11/11/2a8 a", &[]).unwrap();
        // King c8 with attackers b8, d8, c9; the attacker on c1 slides up to c7.
        assert!(matches!(analyse_game(&game), Analysis::Win(m) if m == parse_move("c1c7").unwrap()));
    }

    #[test]
    fn forced_block_of_a_single_escape_route() {
        // King on a6: a3 blocks the way to a1 and b6 its row, so a11 is the only
        // threat, and the attacker on c8 can parry it only by moving to a8.
        let game = Game::from_fen_and_moves("11/11/11/2a8/11/ka9/11/11/a10/9a1/11 a", &[]).unwrap();
        match analyse_game(&game) {
            Analysis::Moves(moves) => assert_eq!(moves.as_slice(), &[parse_move("c8a8").unwrap()]),
            _ => panic!("expected the parrying move"),
        }
        // Without b6, the king also threatens a6k6 with two open corners: lost.
        let game = Game::from_fen_and_moves("11/11/11/2a8/11/k10/11/11/a10/9a1/11 a", &[]).unwrap();
        assert!(matches!(analyse_game(&game), Analysis::Loss));
    }

    /// From self-play: the king on c2 has two open half-rows, to a2 and to k2, each
    /// next to a corner, so the attackers (to move) cannot stop both. They played
    /// b1b2, after which c2k2 wins.
    const C2_KING: &str = "j6j9 g7g10 b6b8 e5b5 d1d2 e7b7 d2a2 g5j5 d11d10 f8h8 j9i9 f5e5 e11e7 g6g8 k4h4                            f6g6 a2a3 g6g3 h1h3 g3c3 f2c2 c3b3 c2b2 d6d9 e1d1 b3c3 a4c4 c3c1 b2b1 c1c2";

    #[test]
    fn escape_in_two_is_a_proven_result() {
        let moves: Vec<&str> = C2_KING.split_whitespace().collect();
        let mut game = Game::from_fen_and_moves(tafl_core::START_FEN, &moves).unwrap();
        assert!(matches!(analyse_game(&game), Analysis::Loss));
        game.try_play(parse_move("b1b2").unwrap()).unwrap();
        assert!(matches!(analyse_game(&game), Analysis::Win(m) if m == parse_move("c2k2").unwrap()));
    }

    #[test]
    fn escape_in_two_is_stopped() {
        // King on c2, hemmed in by b2, c1 and c4: its only decisive move is c2k2
        // (then k1 and k11), and only h5h2 blocks row 2 in time.
        let game = Game::from_fen_and_moves("11/11/11/11/11/11/7a3/2a8/11/1ak8/2a8 a", &[]).unwrap();
        match analyse_game(&game) {
            Analysis::Moves(kept) => assert_eq!(kept.as_slice(), &[parse_move("h5h2").unwrap()]),
            _ => panic!("expected the blocking move"),
        }
        // In the self-play position with a2 occupied, blocking row 2 is not enough:
        // c2c1 captures b1 against the corner and then threatens a1.
        let fen = "5aaa3/3a1ad4/3d4a2/aa4dd2a/ad2ad4a/a3d2d2a/ad2d4da/2a2d1a3/a6a3/a1k8/1a1a1aa4 a";
        let game = Game::from_fen_and_moves(fen, &[]).unwrap();
        assert!(matches!(analyse_game(&game), Analysis::Loss));
    }

    /// The analysis by the plain definitions: every king move and every attacker reply
    /// is tried, in move-generation order.
    fn plain_analysis(pos: &Position, occ: &impl Fn(u64) -> u32) -> Analysis {
        let ends = |next: &Position, side: Side| next.outcome(occ(next.hash) + 1).map(|o| o.winner == side);
        let holds = |p: &Position| {
            p.legal_moves().iter().any(|&mv| {
                let mut next = *p;
                next.play(mv);
                ends(&next, Side::Attackers).unwrap_or_else(|| !king_can_escape(&next))
            })
        };
        let decisive = |p: &Position| {
            p.destinations(p.king).iter().map(|to| Move::new(p.king, to)).find(|&mv| {
                let mut next = *p;
                next.play(mv);
                ends(&next, Side::Defenders).unwrap_or_else(|| king_can_escape(&next) && !holds(&next))
            })
        };
        let fort = |p: &Position| {
            p.legal_moves().iter().copied().find(|&mv| {
                let mut next = *p;
                next.play(mv);
                next.outcome(occ(next.hash) + 1).is_some_and(|o| o.reason == tafl_core::Reason::ExitFort)
            })
        };
        let moves = pos.legal_moves();
        match pos.side {
            Side::Defenders => {
                if let Some(mv) = decisive(pos).or_else(|| fort(pos)) {
                    return Analysis::Win(mv);
                }
                let mut threatened = *pos;
                threatened.set_side(Side::Attackers);
                if king_capture_square(&threatened).is_none() {
                    return Analysis::Moves(moves);
                }
                parry(pos, &moves, occ, |next| king_capture_square(next).is_some())
            }
            Side::Attackers => {
                let repeats = |&mv: &Move| {
                    let mut next = *pos;
                    next.play(mv);
                    next.outcome(occ(next.hash) + 1).is_some_and(|o| o.reason == tafl_core::Reason::Repetition)
                };
                if let Some(mv) = king_capture_move(pos).or_else(|| moves.iter().copied().find(repeats)) {
                    return Analysis::Win(mv);
                }
                let mut threatened = *pos;
                threatened.set_side(Side::Defenders);
                if decisive(&threatened).is_none() && fort(&threatened).is_none() {
                    return Analysis::Moves(moves);
                }
                parry(pos, &moves, occ, |next| decisive(next).is_some() || fort(next).is_some())
            }
        }
    }

    #[test]
    fn analysis_agrees_with_the_plain_definition() {
        let mut rng = tafl_core::rng::Rng::new(11);
        let (mut positions, mut decided) = (0, 0);
        for _ in 0..60 {
            let mut game = Game::standard();
            while !game.is_over() && game.moves().len() < 300 {
                let (fast, plain) = (analyse_game(&game), plain_analysis(game.position(), &|h| game.occurrences(h)));
                positions += 1;
                match (&fast, &plain) {
                    (Analysis::Win(_), Analysis::Win(_)) | (Analysis::Loss, Analysis::Loss) => decided += 1,
                    (Analysis::Moves(a), Analysis::Moves(b)) => assert_eq!(a.as_slice(), b.as_slice()),
                    _ => panic!("analyses differ at ply {}", game.moves().len()),
                }
                let moves = game.position().legal_moves();
                game.try_play(moves.as_slice()[rng.below(moves.len())]).unwrap();
            }
        }
        assert!(positions > 3000 && decided > 50, "{positions} positions, {decided} decided");
    }

    /// From matches against TaflZero, both ending in an exit fort: the defenders' last
    /// move completed it (a defender closing the king's pocket in the first, the king
    /// stepping to the edge in the second).
    const FORTS: [&str; 2] = [
        "j6i6 g7g9 i6i9 e5e3 k6j6 f5c5 g1g3 f4i4 b6b3 h6h3 j6j3 f6f3 e1e2 i4i1 f1g1 f3f4 a6b6 f4j4 k8j8 \
         g5g2 h1h2 i1h1 i9i1 j4i4 k7i7 i4i2 k4k2 i2j2 i1j1 j2i2 g1i1 g6g4 j1j2 i2i6 i7i9 i6k6 j8k8 g4k4 \
         k2k3 f7k7 j3j9 e6j6 a4e4 c5j5",
        "g11g9 f8d8 k7i7 e7e9 b6b9 d6d2 g1g3 e6b6 a4c4 e5e3 a5b5 f5c5 a6a3 e3c3 b5b2 c5c10 a8a10 b6a6 \
         a7a9 f6b6 d1c1 f7a7 a10b10 c10c6 d11c11 g7b7 e1e2 g5b5 c1c2 a6a4 f1c1 c3c5 e11e10 b6a6",
    ];

    #[test]
    fn analysis_agrees_with_the_plain_definition_near_forts() {
        let mut forts = 0;
        for line in FORTS {
            let moves: Vec<&str> = line.split_whitespace().collect();
            for ply in 0..moves.len() {
                let game = Game::from_fen_and_moves(tafl_core::START_FEN, &moves[..ply]).unwrap();
                let (fast, plain) = (analyse_game(&game), plain_analysis(game.position(), &|h| game.occurrences(h)));
                match (&fast, &plain) {
                    (Analysis::Win(_), Analysis::Win(_)) | (Analysis::Loss, Analysis::Loss) => {}
                    (Analysis::Moves(a), Analysis::Moves(b)) => assert_eq!(a.as_slice(), b.as_slice(), "ply {ply}"),
                    _ => panic!("analyses differ at ply {ply}"),
                }
            }
            let game = Game::from_fen_and_moves(tafl_core::START_FEN, &moves).unwrap();
            forts += game.outcome().is_some_and(|o| o.reason == tafl_core::Reason::ExitFort) as usize;
        }
        assert_eq!(forts, FORTS.len());
        // The first fort was closed by a defender other than the king.
        let moves: Vec<&str> = FORTS[0].split_whitespace().collect();
        let game = Game::from_fen_and_moves(tafl_core::START_FEN, &moves[..moves.len() - 1]).unwrap();
        assert!(decisive_king_move(game.position(), &|h| game.occurrences(h), &[]).is_none());
        assert!(matches!(analyse_game(&game), Analysis::Win(_)));
    }

    #[test]
    fn fort_closed_by_a_shieldwall_capture() {
        // c3c1 brackets the attackers on d1 and e1 against the king on f1 (each with a
        // defender on its inner side), and the shieldwall capture leaves the king in a
        // closed pocket d1-f1: e1 was in the pocket but not beside c1.
        let game = Game::from_fen_and_moves("11/9a1/8aa1/11/11/11/11/11/2d8/3ddd5/3aakd4 d", &[]).unwrap();
        let mv = parse_move("c3c1").unwrap();
        assert_eq!(fort_move(game.position(), &|h| game.occurrences(h)), Some(mv));
        let mut next = *game.position();
        next.play(mv);
        assert_eq!(next.outcome(1).map(|o| o.reason), Some(Reason::ExitFort));
    }

    /// Random positions with the king on the edge and defenders crowded around it, so
    /// that forts are common: the filtered search for fort moves finds one exactly when
    /// some move of a defender other than the king completes a fort.
    #[test]
    fn fort_moves_agree_with_the_plain_definition() {
        let mut rng = tafl_core::rng::Rng::new(5);
        let (mut tried, mut found) = (0, 0);
        while tried < 20_000 {
            let king = loop {
                let sq = rng.below(SIZE * SIZE) as Sq;
                if is_edge(sq) && !CORNERS.has(sq) {
                    break sq;
                }
            };
            let mut cells = [b'.'; SIZE * SIZE];
            cells[king as usize] = b'k';
            for (sq, cell) in cells.iter_mut().enumerate() {
                let sq = sq as Sq;
                if *cell != b'.' || tafl_core::geometry::RESTRICTED.has(sq) {
                    continue;
                }
                let near = row_of(sq).abs_diff(row_of(king)).max(col_of(sq).abs_diff(col_of(king))) <= 2;
                let roll = rng.below(100);
                if near && roll < 45 {
                    *cell = b'd';
                } else if roll < if near { 50 } else { 12 } {
                    *cell = b'a';
                }
            }
            let rows: Vec<String> = (0..SIZE)
                .rev()
                .map(|r| {
                    let mut row = String::new();
                    let mut gap = 0;
                    for c in 0..SIZE {
                        match cells[square(r, c) as usize] {
                            b'.' => gap += 1,
                            piece => {
                                if gap > 0 {
                                    row += &gap.to_string();
                                    gap = 0;
                                }
                                row.push(piece as char);
                            }
                        }
                    }
                    if gap > 0 {
                        row += &gap.to_string();
                    }
                    row
                })
                .collect();
            let Ok(game) = Game::from_fen_and_moves(&format!("{} d", rows.join("/")), &[]) else { continue };
            if game.is_over() {
                continue;
            }
            let pos = game.position();
            let occ = |h| game.occurrences(h);
            let plain = pos.legal_moves().iter().any(|&mv| {
                let mut next = *pos;
                next.play(mv);
                mv.from != pos.king && next.outcome(occ(next.hash) + 1).is_some_and(|o| o.reason == Reason::ExitFort)
            });
            assert_eq!(fort_move(pos, &occ).is_some(), plain, "{}", rows.join("/"));
            tried += 1;
            found += plain as usize;
        }
        assert!(found > 500, "only {found} forts in {tried} positions");
    }

    /// From a match against TaflZero: both sides shuffled, and the attackers' e9e8
    /// brought about a position for the third time.
    const SHUFFLE: &str = "g11g9 f8d8 k7i7 f7f9 e11e9 f6f8 k8h8 d8d9 a8c8 f8f9 a7a9 e7e10 d11d10 e6e10 d10b10 d9d11         c8c11 f9b9 f11e11 d6d10 g1g3 g7c7 a5c5 d10d11 e1e4 e10c10 k4g4 e5e9 f10d10 e9d9 h11f11 d9e9 f11f10 e9d9 e4f4         d9d8 f4f9 c7c10 f9c9 d8d9 k5h5 g5e5 h8h7 d9d8 h7h8 e5e9 h5h6 b9b7 d1d7 b7c7 h8e8 c7c6 h6d6 c6c9 c5c8 c9b9         b6b8 b9d9 c8c9 d9d8 a6c6 e9f9 f2f8 d8d9 e8e9 d9d8 e9e8 d8d9 e8e9 d9d8";

    #[test]
    fn third_repetition_is_a_win_for_the_attackers() {
        let moves: Vec<&str> = SHUFFLE.split_whitespace().collect();
        let game = Game::from_fen_and_moves(tafl_core::START_FEN, &moves).unwrap();
        assert!(matches!(analyse_game(&game), Analysis::Win(m) if m == parse_move("e9e8").unwrap()));
        // One move earlier the defenders still had safe moves.
        let game = Game::from_fen_and_moves(tafl_core::START_FEN, &moves[..moves.len() - 1]).unwrap();
        assert!(matches!(analyse_game(&game), Analysis::Moves(_)));
    }

    #[test]
    fn double_escape_threat_is_lost() {
        // King on a6 with both a1 and a11 open; c4 can close only one of them.
        let game = Game::from_fen_and_moves("11/11/11/11/11/k10/11/2a8/11/11/11 a", &[]).unwrap();
        assert!(matches!(analyse_game(&game), Analysis::Loss));
    }
}
