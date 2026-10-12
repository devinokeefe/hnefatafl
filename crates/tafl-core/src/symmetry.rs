//! The eight symmetries of the square board (the dihedral group D4).

use crate::geometry::{NUM_SQUARES, SIZE, Sq, col_of, row_of, square};
use crate::position::{Move, NO_SQUARE, Piece, Position};

pub const NUM_SYMMETRIES: usize = 8;

/// Image of `sq` under symmetry `t` (0 is the identity, 1-3 rotations, 4-7 reflections).
pub const fn transform_square(t: usize, sq: Sq) -> Sq {
    let n = SIZE - 1;
    let (r, c) = (row_of(sq), col_of(sq));
    let (r2, c2) = match t {
        0 => (r, c),
        1 => (c, n - r),
        2 => (n - r, n - c),
        3 => (n - c, r),
        4 => (r, n - c),
        5 => (n - r, c),
        6 => (c, r),
        _ => (n - c, n - r),
    };
    square(r2, c2)
}

/// The symmetry that undoes `t`.
pub const fn inverse(t: usize) -> usize {
    match t {
        1 => 3,
        3 => 1,
        other => other,
    }
}

pub fn transform_move(t: usize, mv: Move) -> Move {
    Move::new(transform_square(t, mv.from), transform_square(t, mv.to))
}

/// The position with every piece (and the last-move square) mapped by `t`.
pub fn transform_position(t: usize, pos: &Position) -> Position {
    let mut out = Position::empty();
    for sq in 0..NUM_SQUARES as Sq {
        if let Some(piece) = pos.piece_at(sq) {
            out.put(transform_square(t, sq), piece);
        }
    }
    out.set_side(pos.side);
    out.last_to = if pos.last_to == NO_SQUARE { NO_SQUARE } else { transform_square(t, pos.last_to) };
    out.last_capture = pos.last_capture;
    debug_assert!(pos.piece_at(pos.king).is_none() || out.piece_at(out.king) == Some(Piece::King));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_laws() {
        for t in 0..NUM_SYMMETRIES {
            for sq in 0..NUM_SQUARES as Sq {
                assert_eq!(transform_square(inverse(t), transform_square(t, sq)), sq);
            }
        }
        // Distinct transforms.
        let probe = square(1, 3);
        let images: std::collections::HashSet<_> = (0..8).map(|t| transform_square(t, probe)).collect();
        assert_eq!(images.len(), 8);
    }

    #[test]
    fn start_position_is_symmetric() {
        let start = Position::start();
        for t in 0..NUM_SYMMETRIES {
            assert_eq!(transform_position(t, &start).fen(), start.fen());
        }
    }
}
