//! A reference for other implementations of the rules: random and tactical games from
//! the `difftest` generator, written as JSON lines with every position's legal moves and
//! network input planes, and each move's captures, repetition count and result.

use std::collections::BTreeMap;
use std::io::Write;

use serde_json::{Value, json};
use tafl_core::{Game, Move, Outcome, Side, square_name};
use tafl_engine::encode::{CompactState, FEATURE_BYTES, HISTORY_MOVES};

use crate::difftest::{Style, choose_move, game_rng, is_shieldwall, start_position};

/// The difftest styles, and edge setups that exercise the shieldwall scan.
const STYLES: [Style; 8] = [
    Style::Uniform,
    Style::Tactical,
    Style::Setup,
    Style::Siege,
    Style::Shuffle,
    Style::Walls,
    Style::Ring,
    Style::Edge,
];

pub struct Options {
    pub games: usize,
    pub seed: u64,
    pub max_plies: usize,
}

#[derive(Default)]
struct Tally {
    positions: u64,
    endings: BTreeMap<String, u64>,
    shieldwalls: u64,
}

/// Input planes of the game's current position, built as self-play builds them.
fn planes(game: &Game) -> Vec<u8> {
    let recent: Vec<Move> = game.moves().iter().rev().take(HISTORY_MOVES).copied().collect();
    let state = CompactState::new(game.position(), game.moves().len(), &recent, |h| game.occurrences(h));
    let mut out = vec![0u8; FEATURE_BYTES];
    state.write_planes(0, &mut out);
    out
}

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().enumerate().fold(0u32, |n, (i, &b)| n | u32::from(b) << (16 - 8 * i));
        for i in 0..4 {
            out.push(if i <= chunk.len() { ALPHABET[(n >> (18 - 6 * i)) as usize & 63] as char } else { '=' });
        }
    }
    out
}

fn outcome_json(outcome: Option<Outcome>) -> Value {
    match outcome {
        Some(o) => json!({
            "winner": if o.winner == Side::Attackers { "attackers" } else { "defenders" },
            "reason": format!("{:?}", o.reason),
        }),
        None => Value::Null,
    }
}

fn sorted_names(names: impl Iterator<Item = String>) -> Vec<String> {
    let mut names: Vec<String> = names.collect();
    names.sort();
    names
}

/// Game `index` of the run, played to its end or to `max_plies`.
fn record_game(seed: u64, index: usize, max_plies: usize, tally: &mut Tally) -> Value {
    let mut rng = game_rng(seed, index);
    let style = STYLES[index % STYLES.len()];
    let mut game = Game::new(start_position(style, &mut rng));
    let mut plies = Vec::new();
    while !game.is_over() && game.moves().len() < max_plies {
        let before = *game.position();
        let legal = sorted_names(before.legal_moves().iter().map(|m| m.to_string()));
        let planes = base64(&planes(&game));
        let mv = choose_move(style, &game, &mut rng);
        let mut after = before;
        let captured = after.play(mv);
        tally.shieldwalls += u64::from(is_shieldwall(mv, captured, &after));
        game.play_unchecked(mv);
        plies.push(json!({
            "fen": before.fen(),
            "legal": legal,
            "move": mv.to_string(),
            "captured": sorted_names(captured.iter().map(square_name)),
            "after": after.fen(),
            "occurrences": game.occurrences(after.hash),
            "outcome": outcome_json(game.outcome()),
            "planes": planes,
        }));
    }
    tally.positions += plies.len() as u64;
    let ending = game.outcome().map_or("ply cap".to_string(), |o| format!("{:?}", o.reason));
    *tally.endings.entry(ending).or_default() += 1;
    json!({
        "seed": seed,
        "game": index,
        "style": format!("{style:?}"),
        "start": game.start().fen(),
        "moves": game.moves().iter().map(|m| m.to_string()).collect::<Vec<_>>(),
        "plies": plies,
    })
}

pub fn run(options: &Options) {
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    let mut tally = Tally::default();
    for index in 0..options.games {
        let line = record_game(options.seed, index, options.max_plies, &mut tally);
        writeln!(out, "{line}").expect("cannot write to stdout");
    }
    out.flush().expect("cannot write to stdout");
    eprintln!("dumped {} positions in {} games", tally.positions, options.games);
    eprintln!("game endings: {:?}", tally.endings);
    eprintln!("moves with shieldwall captures: {}", tally.shieldwalls);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_the_standard() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"M"), "TQ==");
        assert_eq!(base64(b"Ma"), "TWE=");
        assert_eq!(base64(b"Man"), "TWFu");
        assert_eq!(base64(&[0xFB, 0xFF]), "+/8=");
    }

    #[test]
    fn records_replay() {
        let mut tally = Tally::default();
        for index in 0..STYLES.len() {
            let record = record_game(3, index, 60, &mut tally);
            let moves: Vec<&str> = record["moves"].as_array().unwrap().iter().map(|m| m.as_str().unwrap()).collect();
            let game = Game::from_fen_and_moves(record["start"].as_str().unwrap(), &moves).unwrap();
            let plies = record["plies"].as_array().unwrap();
            assert_eq!(plies.len(), moves.len());
            for (i, ply) in plies.iter().enumerate() {
                let next = plies.get(i + 1).map_or(game.position().fen(), |p| p["fen"].as_str().unwrap().to_string());
                assert_eq!(ply["after"].as_str().unwrap(), next);
                assert_eq!(ply["planes"].as_str().unwrap().len(), FEATURE_BYTES.div_ceil(3) * 4);
                assert!(ply["legal"].as_array().unwrap().iter().any(|m| *m == ply["move"]));
            }
            assert_eq!(plies.last().unwrap()["outcome"], outcome_json(game.outcome()));
        }
    }
}
