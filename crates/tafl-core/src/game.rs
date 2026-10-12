//! A game record: the starting position, the moves played and repetition counts.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::geometry::parse_square;
use crate::position::{Move, Position};
use crate::terminal::Outcome;

/// Parses a move such as `a6b6` or `d11d7`.
pub fn parse_move(text: &str) -> Option<Move> {
    let bytes = text.as_bytes();
    // The destination starts at the second letter.
    let split = bytes.iter().skip(1).position(|b| b.is_ascii_lowercase())? + 1;
    Some(Move::new(parse_square(&text[..split])?, parse_square(&text[split..])?))
}

/// Repetition counts by position hash. Zobrist hashes are uniformly random already, so
/// each key serves as its own hash (the default SipHash made lookups a large part of
/// the tactical analysis).
pub type OccurrenceMap = HashMap<u64, u32, BuildHasherDefault<ZobristHasher>>;

#[derive(Default)]
pub struct ZobristHasher(u64);

impl Hasher for ZobristHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(b);
        }
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }
}

#[derive(Clone)]
pub struct Game {
    start: Position,
    position: Position,
    moves: Vec<Move>,
    /// Occurrences of each (board, side to move) after a move. As in TaflZero, the
    /// starting position is not counted.
    occurrences: OccurrenceMap,
    outcome: Option<Outcome>,
}

impl Game {
    pub fn new(start: Position) -> Game {
        Game {
            start,
            position: start,
            moves: Vec::new(),
            occurrences: OccurrenceMap::default(),
            outcome: start.outcome(0),
        }
    }

    pub fn standard() -> Game {
        Game::new(Position::start())
    }

    /// A game from a FEN followed by space-separated moves.
    pub fn from_fen_and_moves(fen: &str, moves: &[&str]) -> Result<Game, String> {
        let mut game = Game::new(Position::from_fen(fen)?);
        for text in moves {
            let mv = parse_move(text).ok_or_else(|| format!("malformed move '{text}'"))?;
            game.try_play(mv)?;
        }
        Ok(game)
    }

    pub fn position(&self) -> &Position {
        &self.position
    }

    pub fn start(&self) -> &Position {
        &self.start
    }

    pub fn moves(&self) -> &[Move] {
        &self.moves
    }

    pub fn outcome(&self) -> Option<Outcome> {
        self.outcome
    }

    pub fn is_over(&self) -> bool {
        self.outcome.is_some()
    }

    /// Times the position with this hash has occurred so far.
    pub fn occurrences(&self, hash: u64) -> u32 {
        self.occurrences.get(&hash).copied().unwrap_or(0)
    }

    pub fn occurrence_map(&self) -> &OccurrenceMap {
        &self.occurrences
    }

    /// Plays a move after checking that the game is running and the move is legal.
    pub fn try_play(&mut self, mv: Move) -> Result<(), String> {
        if let Some(outcome) = self.outcome {
            return Err(format!("the game is over ({outcome:?})"));
        }
        if !self.position.is_legal(mv) {
            return Err(format!("illegal move {mv}"));
        }
        self.play_unchecked(mv);
        Ok(())
    }

    /// Plays a move known to be legal in a running game.
    pub fn play_unchecked(&mut self, mv: Move) {
        self.position.play(mv);
        self.moves.push(mv);
        let count = self.occurrences.entry(self.position.hash).or_insert(0);
        *count += 1;
        self.outcome = self.position.outcome(*count);
    }
}
