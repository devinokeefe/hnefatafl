//! Zobrist keys for repetition detection (fixed at compile time).

use crate::geometry::{NUM_SQUARES, Sq};
use crate::position::Piece;

const fn splitmix64(state: u64) -> (u64, u64) {
    let state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (state, z ^ (z >> 31))
}

const fn build_keys() -> ([[u64; NUM_SQUARES]; 3], u64) {
    let mut keys = [[0u64; NUM_SQUARES]; 3];
    let mut state = 0x7AF1_2026_0000_0011;
    let mut piece = 0;
    while piece < 3 {
        let mut sq = 0;
        while sq < NUM_SQUARES {
            let (next, key) = splitmix64(state);
            state = next;
            keys[piece][sq] = key;
            sq += 1;
        }
        piece += 1;
    }
    let (_, side) = splitmix64(state);
    (keys, side)
}

static KEYS: ([[u64; NUM_SQUARES]; 3], u64) = build_keys();

/// XORed into the hash when the defenders are to move.
pub const SIDE_KEY: u64 = build_keys().1;

#[inline]
pub fn piece_key(piece: Piece, sq: Sq) -> u64 {
    let index = match piece {
        Piece::Attacker => 0,
        Piece::Defender => 1,
        Piece::King => 2,
    };
    KEYS.0[index][sq as usize]
}
