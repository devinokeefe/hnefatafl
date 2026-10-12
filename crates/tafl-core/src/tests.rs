//! Rule tests on hand-made positions. Broad agreement with TaflZero itself is checked
//! separately by `tafl difftest`, which replays random games through its rules probe.

use crate::geometry::*;
use crate::position::*;
use crate::terminal::{Outcome, Reason};
use crate::{Game, parse_move};

/// Builds a position from 11 rows (rank 11 first) of `.`/`+` (empty), `a`, `d`, `k`.
fn diagram(rows: &str, side: Side) -> Position {
    let lines: Vec<&str> = rows.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    assert_eq!(lines.len(), SIZE);
    let mut pos = Position::empty();
    for (i, line) in lines.iter().enumerate() {
        let cells: Vec<char> = line.chars().filter(|c| !c.is_whitespace()).collect();
        assert_eq!(cells.len(), SIZE, "{line}");
        for (col, ch) in cells.into_iter().enumerate() {
            let sq = square(SIZE - 1 - i, col);
            match ch {
                'a' => pos.put(sq, Piece::Attacker),
                'd' => pos.put(sq, Piece::Defender),
                'k' => pos.put(sq, Piece::King),
                '.' | '+' => {}
                _ => panic!("bad cell {ch}"),
            }
        }
    }
    pos.set_side(side);
    pos
}

fn mv(text: &str) -> Move {
    parse_move(text).unwrap()
}

fn sq(name: &str) -> Sq {
    parse_square(name).unwrap()
}

#[test]
fn fen_round_trip_and_start_moves() {
    let start = Position::start();
    assert_eq!(start.fen(), START_FEN);
    assert_eq!(start.attackers.count(), 24);
    assert_eq!(start.defenders.count(), 12);
    assert_eq!(start.king, THRONE);
    assert_eq!(start.legal_moves().len(), 116);
    assert!(Position::from_fen("11/11 a").is_err());
    assert!(Position::from_fen("k10/11/11/11/11/11/11/11/11/11/10k a").is_err());
}

#[test]
fn soldiers_pass_but_never_stop_on_restricted_squares() {
    let pos = diagram(
        "+ . . . . . . . . . +
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         a . . . . + . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         + . . . . . . . . . +",
        Side::Attackers,
    );
    let dest = pos.destinations(sq("a6"));
    assert!(!dest.has(sq("f6")) && dest.has(sq("g6")) && dest.has(sq("k6")));
    assert!(!dest.has(sq("a1")) && !dest.has(sq("a11")) && dest.has(sq("a2")));
}

#[test]
fn king_may_stop_on_corner_and_throne() {
    let pos = diagram(
        ". . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . k
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .",
        Side::Defenders,
    );
    let dest = pos.destinations(sq("k6"));
    assert!(dest.has(sq("k1")) && dest.has(sq("k11")) && dest.has(sq("f6")) && dest.has(sq("a6")));
}

#[test]
fn sandwich_captures() {
    // d4 captured by the attacker arriving on c4 against e4; the king captures too.
    let mut pos = diagram(
        ". . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . a d a . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .",
        Side::Attackers,
    );
    // Moving into a sandwich is safe: the existing a-d-a line stays.
    pos.remove(sq("c4"));
    pos.put(sq("c2"), Piece::Attacker);
    let captured = pos.play(mv("c2c4"));
    assert_eq!(captured, Bb::from_sq(sq("d4")));
    assert!(pos.last_capture);

    let mut safe = diagram(
        ". . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . a . a . . . . . .
         . . . d . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .",
        Side::Defenders,
    );
    assert!(safe.play(mv("d3d4")).is_empty());

    let mut armed = diagram(
        ". . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . d a . . . . . . .
         . . . . k . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .",
        Side::Defenders,
    );
    assert_eq!(armed.play(mv("e3e4")), Bb::from_sq(sq("d4")));
}

#[test]
fn hostile_corners_and_throne() {
    // Corner a1 is hostile to both sides: an attacker arriving on a3 takes a2.
    let mut corner = diagram(
        ". . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         d . . . . . . . . . .
         + . . . . . . . . . .",
        Side::Attackers,
    );
    corner.put(sq("c3"), Piece::Attacker);
    corner.put(sq("k6"), Piece::King);
    assert_eq!(corner.play(mv("c3a3")), Bb::from_sq(sq("a2")));

    // The empty throne is hostile to defenders...
    let mut throne = Position::empty();
    throne.put(sq("f5"), Piece::Defender);
    throne.put(sq("c4"), Piece::Attacker);
    throne.put(sq("b11"), Piece::King);
    throne.play(mv("c4f4"));
    assert_eq!(throne.piece_at(sq("f5")), None);

    // ...but the king on it only threatens attackers.
    let mut occupied = Position::empty();
    occupied.put(sq("f6"), Piece::King);
    occupied.put(sq("f5"), Piece::Defender);
    occupied.put(sq("c4"), Piece::Attacker);
    occupied.play(mv("c4f4"));
    assert_eq!(occupied.piece_at(sq("f5")), Some(Piece::Defender));

    let mut king_throne = Position::empty();
    king_throne.put(sq("f6"), Piece::King);
    king_throne.put(sq("f5"), Piece::Attacker);
    king_throne.put(sq("c4"), Piece::Defender);
    king_throne.set_side(Side::Defenders);
    king_throne.play(mv("c4f4"));
    assert_eq!(king_throne.piece_at(sq("f5")), None);
}

#[test]
fn shieldwall_captures_but_spares_the_king() {
    // Attackers c1 d1 e1 with defenders above; defenders bracket on b1 and f1.
    let mut pos = diagram(
        ". . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . d d d . . . . . .
         . d a a a . d . . . .",
        Side::Defenders,
    );
    pos.put(sq("k6"), Piece::King);
    let captured = pos.play(mv("g1f1"));
    assert_eq!(captured, Bb::from_sq(sq("c1")) | Bb::from_sq(sq("d1")) | Bb::from_sq(sq("e1")));

    // Without the completing move on a bracket nothing happens.
    let mut idle = diagram(
        ". . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . d d d . . . . . .
         . d a a a d . . . . .",
        Side::Defenders,
    );
    idle.put(sq("k6"), Piece::King);
    idle.put(sq("j5"), Piece::Defender);
    assert!(idle.play(mv("j5j4")).is_empty());

    // The defending king inside a wall survives; its partner is taken. Corner a1
    // brackets the run.
    let mut king_wall = diagram(
        ". . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . a a . . . . . . . .
         + k d . a . . . . . .",
        Side::Attackers,
    );
    let captured = king_wall.play(mv("e1d1"));
    assert_eq!(captured, Bb::from_sq(sq("c1")));
    assert_eq!(king_wall.king, sq("b1"));
}

#[test]
fn shieldwall_single_piece_quirk_matches_taflzero() {
    // F E F E E F along rank 1 with friendly roofs; the move to j1 closes the second
    // run. TaflZero keeps the lone first enemy in the run, so it is captured as well.
    let mut pos = diagram(
        ". . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . . . . . . . .
         . . . . d . d d . . .
         . . . d a d a a . . .",
        Side::Defenders,
    );
    pos.put(sq("k6"), Piece::King);
    pos.put(sq("i3"), Piece::Defender);
    let captured = pos.play(mv("i3i1"));
    assert_eq!(captured, Bb::from_sq(sq("e1")) | Bb::from_sq(sq("g1")) | Bb::from_sq(sq("h1")));
}

#[test]
fn king_capture_needs_four_sides() {
    // Three attackers around the king on c8 are not enough...
    let mut three = Position::empty();
    for name in ["c9", "b8"] {
        three.put(sq(name), Piece::Attacker);
    }
    three.put(sq("h8"), Piece::Attacker);
    three.put(sq("c8"), Piece::King);
    three.play(mv("h8d8"));
    assert!(three.has_king());
    // ...the fourth one captures it.
    three.put(sq("h7"), Piece::Attacker);
    three.set_side(Side::Attackers);
    three.play(mv("h7c7"));
    assert!(!three.has_king());
    assert_eq!(three.outcome(1), Some(Outcome { winner: Side::Attackers, reason: Reason::KingCaptured }));

    // Beside the throne three attackers suffice.
    let mut throne = Position::empty();
    throne.put(sq("f7"), Piece::King);
    throne.put(sq("e7"), Piece::Attacker);
    throne.put(sq("g7"), Piece::Attacker);
    throne.put(sq("c8"), Piece::Attacker);
    throne.play(mv("c8f8"));
    assert!(!throne.has_king());

    // On the edge the king cannot be captured.
    let mut edge = Position::empty();
    edge.put(sq("f1"), Piece::King);
    edge.put(sq("e1"), Piece::Attacker);
    edge.put(sq("g1"), Piece::Attacker);
    edge.put(sq("c2"), Piece::Attacker);
    edge.play(mv("c2f2"));
    assert!(edge.has_king());
}

#[test]
fn corner_escape_and_repetition() {
    let mut game = Game::from_fen_and_moves("11/11/11/11/11/11/11/11/11/a10/1k7a1 d", &[]).unwrap();
    game.try_play(mv("b1a1")).unwrap();
    assert_eq!(game.outcome(), Some(Outcome { winner: Side::Defenders, reason: Reason::KingEscaped }));
    assert!(game.try_play(mv("a2b2")).is_err());

    let mut game = Game::standard();
    for _ in 0..2 {
        for m in ["a4b4", "e5d5", "b4a4", "d5e5"] {
            game.try_play(mv(m)).unwrap();
        }
    }
    // The start position itself is not counted, so after two cycles nothing has
    // occurred three times; the next move repeats the first position a third time.
    assert!(!game.is_over());
    game.try_play(mv("a4b4")).unwrap();
    assert_eq!(game.outcome(), Some(Outcome { winner: Side::Attackers, reason: Reason::Repetition }));
}

#[test]
fn encirclement() {
    // A ring of attackers with a gap at f8, closed by the attacker from f11.
    let mut ring = Game::from_fen_and_moves("5a5/11/11/4a1a4/3a3a3/3a1k1a3/3a3a3/4aaa4/11/11/11 a", &[]).unwrap();
    assert!(!ring.is_over());
    ring.try_play(mv("f11f8")).unwrap();
    assert_eq!(ring.outcome(), Some(Outcome { winner: Side::Attackers, reason: Reason::Encircled }));
}

#[test]
fn exit_fort() {
    // King on the edge, walled in by defenders that attackers cannot capture.
    let fort = "11/11/11/11/11/11/11/11/a3ddd4/4d1d4/3adkda3 d";
    let mut game = Game::from_fen_and_moves(fort, &[]).unwrap();
    // Set up directly: not gated, and the fort holds.
    assert_eq!(game.outcome(), Some(Outcome { winner: Side::Defenders, reason: Reason::ExitFort }));
    // A fort with an attacker inside is not a fort.
    let broken = Game::from_fen_and_moves("11/11/11/11/11/11/11/11/4ddd4/4dad4/4dkd4 d", &[]).unwrap();
    assert_eq!(broken.outcome(), None);
    // Nor is a king whose region is open to the attackers.
    let weak = Game::from_fen_and_moves("11/11/11/11/11/11/11/11/11/a3d1d3a/4dkd4 d", &[]).unwrap();
    assert_eq!(weak.outcome(), None);
    game = Game::from_fen_and_moves("11/11/11/11/11/11/11/11/a3d1d4/4d1d4/3adkda3 d", &["e3f3"]).unwrap();
    assert_eq!(game.outcome(), Some(Outcome { winner: Side::Defenders, reason: Reason::ExitFort }));
    // An attacker on a corner, possible only in a set-up position, belongs to no attacker
    // area; the test used to loop for ever on one.
    let cornered = Game::from_fen_and_moves("10a/11/11/11/11/11/11/11/a3ddd4/4d1d4/3adkda3 d", &[]).unwrap();
    assert_eq!(cornered.outcome(), Some(Outcome { winner: Side::Defenders, reason: Reason::ExitFort }));
}

#[test]
fn no_legal_move_loses() {
    // The lone attacker on a2... boxed in: a1 corner, b2 and a3 defenders.
    let game = Game::from_fen_and_moves("11/11/11/11/11/11/11/11/d10/ad3k5/11 a", &[]).unwrap();
    assert_eq!(game.outcome(), Some(Outcome { winner: Side::Defenders, reason: Reason::NoMoves }));
}

#[test]
fn hash_is_incremental() {
    let mut game = Game::standard();
    let mut rng = 0x1234_5678_u64;
    for _ in 0..200 {
        if game.is_over() {
            break;
        }
        let moves = game.position().legal_moves();
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let choice = moves[(rng >> 33) as usize % moves.len()];
        let before = *game.position();
        game.try_play(choice).unwrap();
        if game.position().occupied().count() == before.occupied().count() {
            assert_eq!(before.quiet_hash_after(choice), game.position().hash);
        }
        let fresh = Position::from_fen(&game.position().fen()).unwrap();
        assert_eq!(fresh.hash, game.position().hash);
        assert_eq!(fresh.legal_moves().as_slice(), game.position().legal_moves().as_slice());
    }
}
