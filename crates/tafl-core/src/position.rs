//! Position, move generation and move application (captures).
//!
//! The capture rules reproduce the TaflZero production engine exactly, including its
//! shieldwall scan, because the two engines replay each other's move lists during
//! matches and must never disagree about the resulting board.

use std::fmt;

use crate::geometry::*;
use crate::zobrist;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Side {
    Attackers = 0,
    Defenders = 1,
}

impl Side {
    #[inline]
    pub const fn other(self) -> Side {
        match self {
            Side::Attackers => Side::Defenders,
            Side::Defenders => Side::Attackers,
        }
    }

    pub const fn letter(self) -> char {
        match self {
            Side::Attackers => 'a',
            Side::Defenders => 'd',
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Piece {
    Attacker,
    Defender,
    King,
}

impl Piece {
    pub const fn side(self) -> Side {
        match self {
            Piece::Attacker => Side::Attackers,
            Piece::Defender | Piece::King => Side::Defenders,
        }
    }
}

/// A move from one square to another along a rank or file.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Move {
    pub from: Sq,
    pub to: Sq,
}

impl Move {
    pub const fn new(from: Sq, to: Sq) -> Move {
        Move { from, to }
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", square_name(self.from), square_name(self.to))
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

pub const NO_SQUARE: Sq = u8::MAX;
/// More than any position can have: 24 attackers with at most 20 destinations each.
pub const MAX_MOVES: usize = 512;

/// Fixed-capacity move list (no allocation in the search hot path).
pub struct MoveList {
    moves: [Move; MAX_MOVES],
    len: usize,
}

impl MoveList {
    #[inline]
    pub fn new() -> MoveList {
        MoveList { moves: [Move::default(); MAX_MOVES], len: 0 }
    }

    #[inline]
    pub fn push(&mut self, mv: Move) {
        self.moves[self.len] = mv;
        self.len += 1;
    }

    #[inline]
    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Deref for MoveList {
    type Target = [Move];
    fn deref(&self) -> &[Move] {
        self.as_slice()
    }
}

/// Everything the rules need to know about a position, apart from repetition history.
///
/// `last_to` and `last_capture` describe the move that produced the position: the
/// encirclement and exit-fort tests are only run after moves that can create them,
/// and the TaflZero contract makes that gating part of the rules.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    pub attackers: Bb,
    /// Defenders other than the king.
    pub defenders: Bb,
    /// King square or `NO_SQUARE` once captured.
    pub king: Sq,
    pub side: Side,
    /// Destination of the last move, `NO_SQUARE` for a position set up directly.
    pub last_to: Sq,
    /// Whether the last move captured anything (the king included).
    pub last_capture: bool,
    /// Occupancy in column-major bit order (see [`transposed_bit`]).
    occ_t: u128,
    pub hash: u64,
}

pub const START_FEN: &str = "3aaaaa3/5a5/11/a4d4a/a3ddd3a/aa1ddkdd1aa/a3ddd3a/a4d4a/11/5a5/3aaaaa3 a";

impl Position {
    /// An empty board with attackers to move.
    pub fn empty() -> Position {
        let mut pos = Position {
            attackers: Bb::EMPTY,
            defenders: Bb::EMPTY,
            king: NO_SQUARE,
            side: Side::Attackers,
            last_to: NO_SQUARE,
            last_capture: false,
            occ_t: 0,
            hash: 0,
        };
        pos.hash = pos.compute_hash();
        pos
    }

    pub fn start() -> Position {
        Position::from_fen(START_FEN).expect("start FEN is valid")
    }

    #[inline]
    pub fn king_bb(&self) -> Bb {
        if self.king == NO_SQUARE { Bb::EMPTY } else { Bb::from_sq(self.king) }
    }

    #[inline]
    pub fn has_king(&self) -> bool {
        self.king != NO_SQUARE
    }

    /// Defenders including the king.
    #[inline]
    pub fn defender_side(&self) -> Bb {
        self.defenders | self.king_bb()
    }

    #[inline]
    pub fn occupied(&self) -> Bb {
        self.attackers | self.defenders | self.king_bb()
    }

    #[inline]
    pub fn side_pieces(&self, side: Side) -> Bb {
        match side {
            Side::Attackers => self.attackers,
            Side::Defenders => self.defender_side(),
        }
    }

    pub fn piece_at(&self, sq: Sq) -> Option<Piece> {
        if self.attackers.has(sq) {
            Some(Piece::Attacker)
        } else if self.defenders.has(sq) {
            Some(Piece::Defender)
        } else if self.king == sq {
            Some(Piece::King)
        } else {
            None
        }
    }

    /// Places a piece on an empty square (setup only; leaves `last_to` unchanged).
    pub fn put(&mut self, sq: Sq, piece: Piece) {
        debug_assert!(self.piece_at(sq).is_none());
        match piece {
            Piece::Attacker => self.attackers.set(sq),
            Piece::Defender => self.defenders.set(sq),
            Piece::King => {
                debug_assert!(!self.has_king());
                self.king = sq;
            }
        }
        self.occ_t |= transposed_bit(sq);
        self.hash ^= zobrist::piece_key(piece, sq);
    }

    /// Removes whatever stands on `sq`.
    pub fn remove(&mut self, sq: Sq) {
        if let Some(piece) = self.piece_at(sq) {
            match piece {
                Piece::Attacker => self.attackers.clear(sq),
                Piece::Defender => self.defenders.clear(sq),
                Piece::King => self.king = NO_SQUARE,
            }
            self.occ_t &= !transposed_bit(sq);
            self.hash ^= zobrist::piece_key(piece, sq);
        }
    }

    pub fn set_side(&mut self, side: Side) {
        if self.side != side {
            self.side = side;
            self.hash ^= zobrist::SIDE_KEY;
        }
    }

    fn compute_hash(&self) -> u64 {
        let mut hash = if self.side == Side::Defenders { zobrist::SIDE_KEY } else { 0 };
        for sq in self.attackers {
            hash ^= zobrist::piece_key(Piece::Attacker, sq);
        }
        for sq in self.defenders {
            hash ^= zobrist::piece_key(Piece::Defender, sq);
        }
        if self.has_king() {
            hash ^= zobrist::piece_key(Piece::King, self.king);
        }
        hash
    }

    /// Destination squares of the piece on `from` (which must belong to some side).
    #[inline]
    pub fn destinations(&self, from: Sq) -> Bb {
        let (r, c) = (row_of(from), col_of(from));
        let occ = self.occupied().0;
        let row_occ = ((occ >> (r * SIZE)) & 0x7FF) as usize;
        let col_occ = ((self.occ_t >> (c * SIZE)) & 0x7FF) as usize;
        let mut along_row = LINE[c][row_occ] as u128;
        let mut along_col = LINE[r][col_occ] as u128;
        if from != self.king {
            // Soldiers may cross the empty throne but stop on neither it nor a corner.
            along_row &= !restricted_line_bits(r);
            along_col &= !restricted_line_bits(c);
        }
        let mut dest = along_row << (r * SIZE);
        let mut rows = along_col;
        while rows != 0 {
            let row = rows.trailing_zeros() as usize;
            dest |= 1u128 << (row * SIZE + c);
            rows &= rows - 1;
        }
        Bb(dest)
    }

    pub fn legal_moves(&self) -> MoveList {
        let mut list = MoveList::new();
        self.legal_moves_into(&mut list);
        list
    }

    pub fn legal_moves_into(&self, list: &mut MoveList) {
        list.len = 0;
        for from in self.side_pieces(self.side) {
            for to in self.destinations(from) {
                list.push(Move::new(from, to));
            }
        }
    }

    pub fn has_legal_move(&self) -> bool {
        self.side_pieces(self.side).iter().any(|from| self.destinations(from).any())
    }

    pub fn is_legal(&self, mv: Move) -> bool {
        (mv.from as usize) < NUM_SQUARES
            && self.side_pieces(self.side).has(mv.from)
            && (mv.to as usize) < NUM_SQUARES
            && self.destinations(mv.from).has(mv.to)
    }

    /// Plays a legal move: moves the piece, resolves captures and passes the turn.
    /// Returns the squares whose pieces were captured (the king's included).
    pub fn play(&mut self, mv: Move) -> Bb {
        debug_assert!(self.is_legal(mv), "illegal move {mv:?}");
        let mover = self.side;
        let piece = self.piece_at(mv.from).expect("move from an empty square");
        self.remove(mv.from);
        self.put(mv.to, piece);
        self.last_to = mv.to;

        // Sandwiches are resolved before the shieldwall scan, which sees their result.
        let mut captured = self.sandwich_captures(mv.to, mover);
        for sq in captured {
            self.remove(sq);
        }
        if is_edge(mv.to) {
            let wall = self.shieldwall_captures(mv.to, mover);
            for sq in wall {
                self.remove(sq);
            }
            captured |= wall;
        }
        if mover == Side::Attackers && self.king_is_captured(mv.to) {
            captured.set(self.king);
            self.remove(self.king);
        }

        self.last_capture = captured.any();
        self.side = mover.other();
        self.hash ^= zobrist::SIDE_KEY;
        captured
    }

    /// The hash after `mv`, provided it captures nothing (cheaper than playing it).
    pub fn quiet_hash_after(&self, mv: Move) -> u64 {
        let piece = self.piece_at(mv.from).expect("move from an empty square");
        self.hash ^ zobrist::piece_key(piece, mv.from) ^ zobrist::piece_key(piece, mv.to) ^ zobrist::SIDE_KEY
    }

    /// Squares that are hostile to a soldier of `victim_side` when used as the far
    /// side of a sandwich: the capturing side's pieces and the empty corners/throne.
    #[inline]
    fn hostile_to(&self, victim_side: Side) -> Bb {
        self.side_pieces(victim_side.other()) | (RESTRICTED & !self.occupied())
    }

    /// Ordinary custodian captures made by the piece that just arrived on `to`.
    fn sandwich_captures(&self, to: Sq, mover: Side) -> Bb {
        let victims = match mover {
            Side::Attackers => self.defenders,
            Side::Defenders => self.attackers,
        };
        let hostile = self.hostile_to(mover.other());
        let mut captured = Bb::EMPTY;
        for dir in 0..4 {
            if let (Some(middle), Some(far)) = (step(to, dir, 1), step(to, dir, 2)) {
                if victims.has(middle) && hostile.has(far) {
                    captured.set(middle);
                }
            }
        }
        captured
    }

    /// Shieldwall captures along the edge containing `to`.
    ///
    /// This walks the edge from one corner to the other exactly as TaflZero does: a
    /// run of at least two enemy pieces, each with a friendly piece on its inner side,
    /// bracketed by friendly pieces or corners, is captured when the move just played
    /// is one of the two brackets. The king is never removed by a shieldwall. TaflZero
    /// does not reset a run of exactly one piece at a bracket, so such a piece joins
    /// the next run; that quirk is reproduced so both engines see the same board.
    fn shieldwall_captures(&self, to: Sq, mover: Side) -> Bb {
        let friends = self.side_pieces(mover);
        let enemies = self.side_pieces(mover.other());
        let (r, c) = (row_of(to), col_of(to));
        let last = SIZE - 1;
        // Per edge: the corner it is walked from, the step along it and the inward step, as TaflZero walks it.
        let edges: [(bool, Sq, isize, isize); 4] = [
            (r == last, square(last, 0), 1, -(SIZE as isize)),
            (r == 0, square(0, 0), 1, SIZE as isize),
            (c == 0, square(last, 0), -(SIZE as isize), 1),
            (c == last, square(last, last), -(SIZE as isize), -1),
        ];
        let mut captured = Bb::EMPTY;
        for (on_edge, start, along, inward) in edges {
            if !on_edge {
                continue;
            }
            let mut run = Bb::EMPTY;
            let mut run_len = 0;
            let mut run_open = true;
            let mut bracket = start;
            let mut sq = start;
            for _ in 1..SIZE {
                sq = (sq as isize + along) as Sq;
                let is_far_corner = CORNERS.has(sq);
                if is_far_corner || friends.has(sq) {
                    if run_len > 1 {
                        if to == bracket || to == sq {
                            captured |= run;
                        }
                        run = Bb::EMPTY;
                        run_len = 0;
                    }
                    run_open = true;
                    bracket = sq;
                } else if !enemies.has(sq) {
                    run = Bb::EMPTY;
                    run_len = 0;
                    run_open = false;
                } else if run_open {
                    if friends.has((sq as isize + inward) as Sq) {
                        run.set(sq);
                        run_len += 1;
                    } else {
                        run = Bb::EMPTY;
                        run_len = 0;
                        run_open = false;
                    }
                }
            }
        }
        captured & !self.king_bb()
    }

    /// Whether the attacker that just arrived on `to` completes a capture of the king:
    /// it must be adjacent, and all four sides must be attackers or the throne.
    fn king_is_captured(&self, to: Sq) -> bool {
        if !self.has_king() {
            return false;
        }
        let around = Bb::from_sq(self.king).neighbors();
        around.has(to) && around.count() == 4 && (around & !(self.attackers | THRONE_BB)).is_empty()
    }

    /// Board and side to move in TaflZero FEN notation.
    pub fn fen(&self) -> String {
        let mut out = String::new();
        for row in (0..SIZE).rev() {
            let mut empty = 0;
            for col in 0..SIZE {
                let ch = match self.piece_at(square(row, col)) {
                    Some(Piece::Attacker) => 'a',
                    Some(Piece::Defender) => 'd',
                    Some(Piece::King) => 'k',
                    None => {
                        empty += 1;
                        continue;
                    }
                };
                if empty > 0 {
                    out.push_str(&empty.to_string());
                    empty = 0;
                }
                out.push(ch);
            }
            if empty > 0 {
                out.push_str(&empty.to_string());
            }
            if row > 0 {
                out.push('/');
            }
        }
        out.push(' ');
        out.push(self.side.letter());
        out
    }

    /// Parses a TaflZero FEN (`<rows from rank 11 down> <a|d>`). The result has no
    /// last move, so encirclement and fort tests are not gated.
    pub fn from_fen(fen: &str) -> Result<Position, String> {
        let mut parts = fen.split_whitespace();
        let rows_part = parts.next().ok_or("empty FEN")?;
        let side = match parts.next() {
            Some("a") => Side::Attackers,
            Some("d") => Side::Defenders,
            Some(other) => return Err(format!("unknown side to move '{other}'")),
            None => return Err("FEN lacks the side to move".into()),
        };
        if parts.next().is_some() {
            return Err("trailing text after the side to move".into());
        }
        let rows: Vec<&str> = rows_part.split('/').collect();
        if rows.len() != SIZE {
            return Err(format!("expected {SIZE} rows, found {}", rows.len()));
        }
        let mut pos = Position::empty();
        for (i, text) in rows.iter().enumerate() {
            let row = SIZE - 1 - i;
            let mut col = 0usize;
            let mut run = 0usize;
            for ch in text.chars() {
                if let Some(d) = ch.to_digit(10) {
                    if run == 0 && d == 0 {
                        return Err(format!("row {}: empty run starts with 0", row + 1));
                    }
                    run = run * 10 + d as usize;
                    continue;
                }
                col += run;
                run = 0;
                let piece = match ch {
                    'a' => Piece::Attacker,
                    'd' => Piece::Defender,
                    'k' => Piece::King,
                    _ => return Err(format!("invalid character '{ch}'")),
                };
                if col >= SIZE {
                    return Err(format!("row {} is wider than {SIZE}", row + 1));
                }
                if piece == Piece::King && pos.has_king() {
                    return Err("more than one king".into());
                }
                pos.put(square(row, col), piece);
                col += 1;
            }
            col += run;
            if col != SIZE {
                return Err(format!("row {} has width {col}, expected {SIZE}", row + 1));
            }
        }
        pos.set_side(side);
        Ok(pos)
    }
}

/// Corner/throne bits on line `index` (a rank or file), which soldiers may not stop on.
#[inline]
const fn restricted_line_bits(index: usize) -> u128 {
    if index == CENTER {
        1 << CENTER
    } else if index == 0 || index == SIZE - 1 {
        1 | (1 << (SIZE - 1))
    } else {
        0
    }
}

impl fmt::Debug for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in (0..SIZE).rev() {
            write!(f, "{:>2} ", row + 1)?;
            for col in 0..SIZE {
                let sq = square(row, col);
                let ch = match self.piece_at(sq) {
                    Some(Piece::Attacker) => 'a',
                    Some(Piece::Defender) => 'd',
                    Some(Piece::King) => 'K',
                    None if RESTRICTED.has(sq) => '+',
                    None => '.',
                };
                write!(f, "{ch} ")?;
            }
            writeln!(f)?;
        }
        writeln!(f, "   a b c d e f g h i j k   {:?} to move", self.side)
    }
}
