//! Neural-network interface: input planes and the 4840-entry action space.
//!
//! An action is `plane * 121 + from`, where `plane = direction * 10 + distance - 1`
//! (directions north, south, east, west), matching a policy head that emits 40 planes
//! over the 11x11 board.

use tafl_core::geometry::{CORNERS, RESTRICTED, SIZE, col_of, row_of};
use tafl_core::symmetry::{NUM_SYMMETRIES, transform_move};
use tafl_core::{Bb, Move, NUM_SQUARES, Position, Side, Sq};

pub const NUM_ACTIONS: usize = 40 * NUM_SQUARES;
pub const HISTORY_MOVES: usize = 4;

/// Input planes, each 121 bytes (row-major from a1).
pub mod plane {
    pub const ATTACKERS: usize = 0;
    pub const DEFENDERS: usize = 1;
    pub const KING: usize = 2;
    pub const EMPTY_RESTRICTED: usize = 3;
    pub const DEFENDERS_TO_MOVE: usize = 4;
    pub const ONES: usize = 5;
    /// From/to squares of the last four moves, most recent first.
    pub const HISTORY: usize = 6;
    /// Set everywhere when this position has already occurred twice.
    pub const SECOND_OCCURRENCE: usize = 14;
    /// Destinations of moves that would repeat a position a third time (ending the game).
    pub const REPEAT_THIRD: usize = 15;
    /// Destinations of moves that would repeat a position a second time.
    pub const REPEAT_SECOND: usize = 16;
    pub const KING_REACH_1: usize = 17;
    pub const KING_REACH_2: usize = 18;
    /// Ply number, saturating at 255 (scaled by the network).
    pub const PLY: usize = 19;
    pub const COUNT: usize = 20;
}

pub const NUM_PLANES: usize = plane::COUNT;
pub const FEATURE_BYTES: usize = NUM_PLANES * NUM_SQUARES;

pub fn action_index(mv: Move) -> usize {
    let (fr, fc) = (row_of(mv.from) as isize, col_of(mv.from) as isize);
    let (tr, tc) = (row_of(mv.to) as isize, col_of(mv.to) as isize);
    let (dir, dist) = if tc == fc {
        if tr > fr { (0, tr - fr) } else { (1, fr - tr) }
    } else if tc > fc {
        (2, tc - fc)
    } else {
        (3, fc - tc)
    };
    debug_assert!(dist >= 1 && dist < SIZE as isize && (tr == fr || tc == fc));
    (dir * 10 + dist as usize - 1) * NUM_SQUARES + mv.from as usize
}

/// The move for an action index, if it stays on the board.
pub fn action_move(action: usize) -> Option<Move> {
    let plane = action / NUM_SQUARES;
    let from = (action % NUM_SQUARES) as Sq;
    let (dir, dist) = (plane / 10, (plane % 10 + 1) as isize);
    tafl_core::geometry::step(from, dir, dist).map(|to| Move::new(from, to))
}

/// `table[t][a]`: the action that `a` becomes under symmetry `t` (or `a` itself for
/// off-board actions, which never carry probability).
pub fn symmetry_action_table() -> Vec<[u16; NUM_ACTIONS]> {
    (0..NUM_SYMMETRIES)
        .map(|t| {
            let mut table = [0u16; NUM_ACTIONS];
            for (a, slot) in table.iter_mut().enumerate() {
                *slot = match action_move(a) {
                    Some(mv) => action_index(transform_move(t, mv)) as u16,
                    None => a as u16,
                };
            }
            table
        })
        .collect()
}

/// Everything needed to rebuild the input planes of a position without its game.
/// Stored with every training sample.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct CompactState {
    pub attackers: u128,
    pub defenders: u128,
    pub repeat_third: u128,
    pub repeat_second: u128,
    pub king: u8,
    pub side: u8,
    pub second_occurrence: u8,
    pub ply: u8,
    /// (from, to) of the last moves, most recent first; 255 when absent.
    pub history: [[u8; 2]; HISTORY_MOVES],
    pub _pad: [u8; 4],
}

pub const COMPACT_STATE_BYTES: usize = std::mem::size_of::<CompactState>();

impl CompactState {
    /// `recent` lists the last moves, most recent first; `count` gives how many times a
    /// position hash has occurred so far (game history plus search path).
    pub fn new(pos: &Position, ply: usize, recent: &[Move], count: impl Fn(u64) -> u32) -> CompactState {
        let mut history = [[u8::MAX; 2]; HISTORY_MOVES];
        for (slot, mv) in history.iter_mut().zip(recent) {
            *slot = [mv.from, mv.to];
        }
        let (repeat_third, repeat_second) = repetition_targets(pos, &count);
        CompactState {
            attackers: pos.attackers.0,
            defenders: pos.defenders.0,
            repeat_third: repeat_third.0,
            repeat_second: repeat_second.0,
            king: pos.king,
            side: pos.side as u8,
            second_occurrence: u8::from(count(pos.hash) >= 2),
            ply: ply.min(255) as u8,
            history,
            _pad: [0; 4],
        }
    }

    pub fn position(&self) -> Position {
        let mut pos = Position::empty();
        for sq in Bb(self.attackers) {
            pos.put(sq, tafl_core::Piece::Attacker);
        }
        for sq in Bb(self.defenders) {
            pos.put(sq, tafl_core::Piece::Defender);
        }
        if self.king != tafl_core::NO_SQUARE {
            pos.put(self.king, tafl_core::Piece::King);
        }
        pos.set_side(if self.side == 0 { Side::Attackers } else { Side::Defenders });
        pos
    }

    /// Writes the planes (as 0/1 bytes, the ply plane as a count) for symmetry `t`.
    pub fn write_planes(&self, t: usize, out: &mut [u8]) {
        debug_assert_eq!(out.len(), FEATURE_BYTES);
        out.fill(0);
        let map = |sq: Sq| tafl_core::symmetry::transform_square(t, sq) as usize;
        let mut fill = |plane: usize, bb: Bb| {
            for sq in bb {
                out[plane * NUM_SQUARES + map(sq)] = 1;
            }
        };
        let pos = self.position();
        fill(plane::ATTACKERS, pos.attackers);
        fill(plane::DEFENDERS, pos.defenders);
        fill(plane::KING, pos.king_bb());
        fill(plane::EMPTY_RESTRICTED, RESTRICTED & !pos.occupied());
        fill(plane::REPEAT_THIRD, Bb(self.repeat_third));
        fill(plane::REPEAT_SECOND, Bb(self.repeat_second));
        if pos.has_king() {
            let (one, two) = king_reach(&pos);
            fill(plane::KING_REACH_1, one);
            fill(plane::KING_REACH_2, two);
        }
        for (i, [from, to]) in self.history.iter().enumerate() {
            if *from != u8::MAX {
                out[(plane::HISTORY + 2 * i) * NUM_SQUARES + map(*from)] = 1;
                out[(plane::HISTORY + 2 * i + 1) * NUM_SQUARES + map(*to)] = 1;
            }
        }
        let constant = |out: &mut [u8], plane: usize, value: u8| {
            out[plane * NUM_SQUARES..(plane + 1) * NUM_SQUARES].fill(value);
        };
        if self.side == Side::Defenders as u8 {
            constant(out, plane::DEFENDERS_TO_MOVE, 1);
        }
        constant(out, plane::ONES, 1);
        if self.second_occurrence != 0 {
            constant(out, plane::SECOND_OCCURRENCE, 1);
        }
        constant(out, plane::PLY, self.ply);
    }
}

/// Squares the king reaches in one move, and in exactly two (with the board as is).
pub fn king_reach(pos: &Position) -> (Bb, Bb) {
    let one = pos.destinations(pos.king);
    let mut two = Bb::EMPTY;
    let mut moved = *pos;
    let king = pos.king;
    for sq in one {
        // Slide the king without captures: only occupancy matters for reach.
        moved.remove(moved.king);
        moved.put(sq, tafl_core::Piece::King);
        two |= moved.destinations(sq);
        moved.remove(sq);
        moved.put(king, tafl_core::Piece::King);
    }
    (one, two & !one & !Bb::from_sq(king))
}

/// Destinations of non-capturing moves that would recreate a position seen twice
/// (third occurrence) or once (second occurrence). A capture can never repeat a
/// position, because material only decreases.
pub fn repetition_targets(pos: &Position, count: &impl Fn(u64) -> u32) -> (Bb, Bb) {
    let mut third = Bb::EMPTY;
    let mut second = Bb::EMPTY;
    for mv in pos.legal_moves().iter() {
        let mut next = *pos;
        if next.play(*mv).any() {
            continue;
        }
        match count(next.hash) {
            0 => {}
            1 => second.set(mv.to),
            _ => third.set(mv.to),
        }
    }
    (third, second)
}

/// Whether the king (defenders to move) can step onto a corner right now.
#[inline]
pub fn king_can_escape(pos: &Position) -> bool {
    pos.has_king() && (pos.destinations(pos.king) & CORNERS).any()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tafl_core::symmetry::transform_position;

    #[test]
    fn actions_round_trip() {
        let mut seen = 0;
        for a in 0..NUM_ACTIONS {
            if let Some(mv) = action_move(a) {
                assert_eq!(action_index(mv), a);
                seen += 1;
            }
        }
        // Every square has 20 orthogonal destinations on an empty 11x11 board.
        assert_eq!(seen, NUM_SQUARES * 20);
        let table = symmetry_action_table();
        let pos = Position::start();
        for (t, map) in table.iter().enumerate() {
            let image = transform_position(t, &pos);
            let mut mapped: Vec<usize> = pos.legal_moves().iter().map(|m| map[action_index(*m)] as usize).collect();
            let mut direct: Vec<usize> = image.legal_moves().iter().map(|m| action_index(*m)).collect();
            mapped.sort();
            direct.sort();
            assert_eq!(mapped, direct);
        }
    }

    #[test]
    fn planes_transform_with_the_board() {
        let mut game = tafl_core::Game::standard();
        for m in ["d11d9", "f4b4", "a4a2", "g7g10"] {
            game.try_play(tafl_core::parse_move(m).unwrap()).unwrap();
        }
        let recent: Vec<Move> = game.moves().iter().rev().copied().collect();
        let state = CompactState::new(game.position(), game.moves().len(), &recent, |h| game.occurrences(h));
        let mut base = vec![0u8; FEATURE_BYTES];
        state.write_planes(0, &mut base);
        for t in 0..NUM_SYMMETRIES {
            let mut out = vec![0u8; FEATURE_BYTES];
            state.write_planes(t, &mut out);
            for p in 0..NUM_PLANES {
                for sq in 0..NUM_SQUARES as Sq {
                    let image = tafl_core::symmetry::transform_square(t, sq) as usize;
                    assert_eq!(out[p * NUM_SQUARES + image], base[p * NUM_SQUARES + sq as usize]);
                }
            }
        }
        assert_eq!(state.position().fen(), game.position().fen());
    }

    #[test]
    fn repetition_planes_flag_returning_moves() {
        let mut game = tafl_core::Game::standard();
        for m in ["a4b4", "e5d5", "b4a4", "d5e5", "a4b4", "e5d5", "b4a4"] {
            game.try_play(tafl_core::parse_move(m).unwrap()).unwrap();
        }
        // Defenders to move; d5e5 would recreate the start position for the second time
        // (the start itself is not counted), and the position after "d5e5" occurred once.
        let (third, second) = repetition_targets(game.position(), &|h| game.occurrences(h));
        let e5 = tafl_core::parse_square("e5").unwrap();
        assert!(second.has(e5) && !third.has(e5));
        game.try_play(tafl_core::parse_move("d5e5").unwrap()).unwrap();
        // Attackers: a4b4 would now be a third occurrence (an immediate win).
        let (third, _) = repetition_targets(game.position(), &|h| game.occurrences(h));
        assert!(third.has(tafl_core::parse_square("b4").unwrap()));
    }
}
