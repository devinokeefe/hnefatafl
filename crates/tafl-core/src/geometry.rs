//! Board geometry: squares, 121-bit bitboards, edge/corner masks and sliding tables.
//!
//! Square `row * 11 + col`; row 0 is rank 1 and col 0 is file `a`, so `a1` is square 0
//! and `k11` is square 120 (the TaflZero numbering, which keeps FENs and move strings
//! interchangeable between the two engines).

use std::fmt;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

pub const SIZE: usize = 11;
pub const NUM_SQUARES: usize = SIZE * SIZE;
pub const CENTER: usize = SIZE / 2;

/// A square index in `0..121`.
pub type Sq = u8;

pub const THRONE: Sq = (CENTER * SIZE + CENTER) as Sq;
pub const CORNER_SQUARES: [Sq; 4] = [0, (SIZE - 1) as Sq, ((SIZE - 1) * SIZE) as Sq, (NUM_SQUARES - 1) as Sq];

#[inline]
pub const fn row_of(sq: Sq) -> usize {
    sq as usize / SIZE
}

#[inline]
pub const fn col_of(sq: Sq) -> usize {
    sq as usize % SIZE
}

#[inline]
pub const fn square(row: usize, col: usize) -> Sq {
    (row * SIZE + col) as Sq
}

#[inline]
pub const fn is_edge(sq: Sq) -> bool {
    let (r, c) = (row_of(sq), col_of(sq));
    r == 0 || r == SIZE - 1 || c == 0 || c == SIZE - 1
}

/// Algebraic name such as `a1` or `k11`.
pub fn square_name(sq: Sq) -> String {
    format!("{}{}", (b'a' + col_of(sq) as u8) as char, row_of(sq) + 1)
}

/// Parses `a1`..`k11`; `None` for anything else.
pub fn parse_square(s: &str) -> Option<Sq> {
    let bytes = s.as_bytes();
    let file = *bytes.first()?;
    if !(b'a'..b'a' + SIZE as u8).contains(&file) {
        return None;
    }
    let rank: usize = s[1..].parse().ok()?;
    if !(1..=SIZE).contains(&rank) || s[1..].starts_with('0') {
        return None;
    }
    Some(square(rank - 1, (file - b'a') as usize))
}

/// A set of squares; bit `i` is square `i`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Bb(pub u128);

impl Bb {
    pub const EMPTY: Bb = Bb(0);
    pub const ALL: Bb = Bb((1u128 << NUM_SQUARES) - 1);

    #[inline]
    pub const fn from_sq(sq: Sq) -> Bb {
        Bb(1u128 << sq)
    }

    #[inline]
    pub const fn has(self, sq: Sq) -> bool {
        (self.0 >> sq) & 1 != 0
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub const fn any(self) -> bool {
        self.0 != 0
    }

    #[inline]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Lowest square; only meaningful for a non-empty set.
    #[inline]
    pub const fn first(self) -> Sq {
        self.0.trailing_zeros() as Sq
    }

    #[inline]
    pub fn set(&mut self, sq: Sq) {
        self.0 |= 1u128 << sq;
    }

    #[inline]
    pub fn clear(&mut self, sq: Sq) {
        self.0 &= !(1u128 << sq);
    }

    #[inline]
    pub const fn north(self) -> Bb {
        Bb((self.0 << SIZE) & Bb::ALL.0)
    }

    #[inline]
    pub const fn south(self) -> Bb {
        Bb(self.0 >> SIZE)
    }

    #[inline]
    pub const fn east(self) -> Bb {
        Bb((self.0 << 1) & !FILE_A.0 & Bb::ALL.0)
    }

    #[inline]
    pub const fn west(self) -> Bb {
        Bb((self.0 >> 1) & !FILE_K.0)
    }

    /// Orthogonal neighbours of every square in the set.
    #[inline]
    pub const fn neighbors(self) -> Bb {
        Bb(self.north().0 | self.south().0 | self.east().0 | self.west().0)
    }

    /// Squares reachable from `seed` by orthogonal steps inside `passable`
    /// (seed squares outside `passable` are dropped).
    pub fn flood(seed: Bb, passable: Bb) -> Bb {
        let mut filled = seed & passable;
        loop {
            let grown = filled | (filled.neighbors() & passable);
            if grown == filled {
                return filled;
            }
            filled = grown;
        }
    }

    #[inline]
    pub fn iter(self) -> BbIter {
        BbIter(self.0)
    }
}

pub struct BbIter(u128);

impl Iterator for BbIter {
    type Item = Sq;

    #[inline]
    fn next(&mut self) -> Option<Sq> {
        if self.0 == 0 {
            return None;
        }
        let sq = self.0.trailing_zeros() as Sq;
        self.0 &= self.0 - 1;
        Some(sq)
    }
}

impl IntoIterator for Bb {
    type Item = Sq;
    type IntoIter = BbIter;

    fn into_iter(self) -> BbIter {
        self.iter()
    }
}

macro_rules! bb_binop {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident, $op:tt) => {
        impl $trait for Bb {
            type Output = Bb;
            #[inline]
            fn $method(self, rhs: Bb) -> Bb {
                Bb(self.0 $op rhs.0)
            }
        }
        impl $assign_trait for Bb {
            #[inline]
            fn $assign_method(&mut self, rhs: Bb) {
                self.0 = self.0 $op rhs.0;
            }
        }
    };
}

bb_binop!(BitAnd, bitand, BitAndAssign, bitand_assign, &);
bb_binop!(BitOr, bitor, BitOrAssign, bitor_assign, |);
bb_binop!(BitXor, bitxor, BitXorAssign, bitxor_assign, ^);

impl Not for Bb {
    type Output = Bb;
    #[inline]
    fn not(self) -> Bb {
        Bb(!self.0 & Bb::ALL.0)
    }
}

impl fmt::Debug for Bb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in (0..SIZE).rev() {
            for col in 0..SIZE {
                f.write_str(if self.has(square(row, col)) { "x" } else { "." })?;
            }
            f.write_str("\n")?;
        }
        Ok(())
    }
}

const fn file_mask(col: usize) -> Bb {
    let mut bits = 0u128;
    let mut row = 0;
    while row < SIZE {
        bits |= 1u128 << (row * SIZE + col);
        row += 1;
    }
    Bb(bits)
}

const fn rank_mask(row: usize) -> Bb {
    Bb(((1u128 << SIZE) - 1) << (row * SIZE))
}

pub const FILE_A: Bb = file_mask(0);
pub const FILE_K: Bb = file_mask(SIZE - 1);
pub const RANK_1: Bb = rank_mask(0);
pub const RANK_11: Bb = rank_mask(SIZE - 1);
pub const EDGES: Bb = Bb(FILE_A.0 | FILE_K.0 | RANK_1.0 | RANK_11.0);
pub const CORNERS: Bb = Bb((1u128 << CORNER_SQUARES[0])
    | (1u128 << CORNER_SQUARES[1])
    | (1u128 << CORNER_SQUARES[2])
    | (1u128 << CORNER_SQUARES[3]));
pub const THRONE_BB: Bb = Bb::from_sq(THRONE);
/// Squares only the king may stop on.
pub const RESTRICTED: Bb = Bb(CORNERS.0 | THRONE_BB.0);

/// `LINE[pos][occupancy]`: positions reachable by sliding from `pos` along an 11-square
/// line whose occupied positions are the bits of `occupancy` (the bit at `pos` is ignored).
pub static LINE: [[u16; 1 << SIZE]; SIZE] = build_line_table();

const fn build_line_table() -> [[u16; 1 << SIZE]; SIZE] {
    let mut table = [[0u16; 1 << SIZE]; SIZE];
    let mut pos = 0;
    while pos < SIZE {
        let mut occ = 0;
        while occ < (1 << SIZE) {
            let mut mask = 0u16;
            let mut i = pos as isize - 1;
            while i >= 0 && (occ >> i) & 1 == 0 {
                mask |= 1 << i;
                i -= 1;
            }
            let mut j = pos + 1;
            while j < SIZE && (occ >> j) & 1 == 0 {
                mask |= 1 << j;
                j += 1;
            }
            table[pos][occ] = mask;
            occ += 1;
        }
        pos += 1;
    }
    table
}

/// Bit of square `sq` in the transposed (column-major) layout, used to read a file's
/// occupancy as one contiguous 11-bit field.
#[inline]
pub const fn transposed_bit(sq: Sq) -> u128 {
    1u128 << (col_of(sq) * SIZE + row_of(sq))
}

/// The 8 orthogonal and diagonal neighbours of each square.
pub static KING_RING: [Bb; NUM_SQUARES] = build_rings();

const fn build_rings() -> [Bb; NUM_SQUARES] {
    let mut rings = [Bb::EMPTY; NUM_SQUARES];
    let mut sq = 0;
    while sq < NUM_SQUARES {
        let (r, c) = ((sq / SIZE) as isize, (sq % SIZE) as isize);
        let mut bits = 0u128;
        let mut dr = -1;
        while dr <= 1 {
            let mut dc = -1;
            while dc <= 1 {
                let (nr, nc) = (r + dr, c + dc);
                if (dr != 0 || dc != 0) && nr >= 0 && nr < SIZE as isize && nc >= 0 && nc < SIZE as isize {
                    bits |= 1u128 << (nr as usize * SIZE + nc as usize);
                }
                dc += 1;
            }
            dr += 1;
        }
        rings[sq] = Bb(bits);
        sq += 1;
    }
    rings
}

/// Orthogonal directions as (row delta, col delta): north, south, east, west.
pub const DIRECTIONS: [(isize, isize); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

/// The square `steps` away from `sq` in direction `dir`, if on the board.
#[inline]
pub const fn step(sq: Sq, dir: usize, steps: isize) -> Option<Sq> {
    let r = row_of(sq) as isize + DIRECTIONS[dir].0 * steps;
    let c = col_of(sq) as isize + DIRECTIONS[dir].1 * steps;
    if r >= 0 && r < SIZE as isize && c >= 0 && c < SIZE as isize { Some(square(r as usize, c as usize)) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_names_round_trip() {
        for sq in 0..NUM_SQUARES as Sq {
            assert_eq!(parse_square(&square_name(sq)), Some(sq));
        }
        assert_eq!(square_name(0), "a1");
        assert_eq!(square_name(120), "k11");
        assert_eq!(square_name(THRONE), "f6");
        for bad in ["", "a", "a0", "a12", "l1", "a01", "A1", "a1x"] {
            assert_eq!(parse_square(bad), None, "{bad}");
        }
    }

    #[test]
    fn shifts_do_not_wrap() {
        assert_eq!(Bb::from_sq(square(3, 10)).east(), Bb::EMPTY);
        assert_eq!(Bb::from_sq(square(3, 0)).west(), Bb::EMPTY);
        assert_eq!(Bb::from_sq(square(10, 4)).north(), Bb::EMPTY);
        assert_eq!(Bb::from_sq(square(0, 4)).south(), Bb::EMPTY);
        assert_eq!(Bb::from_sq(THRONE).neighbors().count(), 4);
        assert_eq!(Bb::from_sq(0).neighbors().count(), 2);
    }

    #[test]
    fn flood_respects_walls() {
        let wall = Bb(file_mask(3).0);
        let filled = Bb::flood(Bb::from_sq(0), !wall);
        assert_eq!(filled.count(), 33);
        assert!(!filled.has(square(5, 4)));
    }

    #[test]
    fn line_table_matches_definition() {
        assert_eq!(LINE[5][0], 0b111_1101_1111);
        assert_eq!(LINE[0][0b10], 0);
        assert_eq!(LINE[5][(1 << 3) | (1 << 8)], 0b000_1101_0000);
    }

    #[test]
    fn masks() {
        assert_eq!(EDGES.count(), 40);
        assert_eq!(CORNERS.count(), 4);
        assert_eq!(KING_RING[0].count(), 3);
        assert_eq!(KING_RING[THRONE as usize].count(), 8);
    }
}
