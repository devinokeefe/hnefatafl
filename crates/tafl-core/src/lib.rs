//! Copenhagen Hnefatafl (11x11) rules under the `taflzero-copenhagen-operational-v1`
//! contract used by the TaflZero production engine: strong armed king, hostile empty
//! throne and corners, shieldwalls, edge exit forts, encirclement, and a third
//! occurrence of any position winning for the attackers.

pub mod game;
pub mod geometry;
pub mod position;
pub mod rng;
pub mod symmetry;
pub mod terminal;
mod zobrist;

pub use game::{Game, OccurrenceMap, parse_move};
pub use geometry::{Bb, NUM_SQUARES, SIZE, Sq, parse_square, square, square_name};
pub use position::{MAX_MOVES, Move, MoveList, NO_SQUARE, Piece, Position, START_FEN, Side};
pub use terminal::{Outcome, REPETITION_LIMIT, Reason};

#[cfg(test)]
mod tests;
