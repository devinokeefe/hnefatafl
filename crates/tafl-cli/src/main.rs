//! `tafl`: perft, rules differential testing against TaflZero, a reference dump of the
//! rules, and rules benchmarks.

mod difftest;
mod dump;

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand};
use tafl_core::{Game, Position};

#[derive(Parser)]
#[command(name = "tafl", about = "Copenhagen Hnefatafl engine tools")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Count leaf positions of the move tree (games end at terminal positions).
    Perft {
        #[arg(long, default_value_t = 3)]
        depth: u32,
        #[arg(long)]
        fen: Option<String>,
    },
    /// Compare our rules with TaflZero's on random games, ply by ply.
    Difftest {
        /// TaflZero research probe executable (JSON lines on stdin/stdout).
        #[arg(long)]
        probe: PathBuf,
        #[arg(long, default_value_t = 300)]
        games: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long, default_value_t = 8)]
        workers: usize,
        #[arg(long, default_value_t = 400)]
        max_plies: usize,
    },
    /// Write the difftest games as JSON lines (positions, legal moves, captures,
    /// repetition counts, results, input planes): a reference for other implementations.
    Dump {
        #[arg(long, default_value_t = 200)]
        games: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long, default_value_t = 400)]
        max_plies: usize,
    },
}

fn perft(game: &Game, depth: u32) -> u64 {
    if depth == 0 || game.is_over() {
        return 1;
    }
    let moves = game.position().legal_moves();
    if depth == 1 {
        return moves.len() as u64;
    }
    moves
        .iter()
        .map(|&mv| {
            let mut child = game.clone();
            child.play_unchecked(mv);
            perft(&child, depth - 1)
        })
        .sum()
}

fn main() -> ExitCode {
    match Cli::parse().command {
        Cmd::Perft { depth, fen } => {
            let start = match fen {
                Some(f) => match Position::from_fen(&f) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("bad FEN: {e}");
                        return ExitCode::FAILURE;
                    }
                },
                None => Position::start(),
            };
            for d in 1..=depth {
                let t = Instant::now();
                let n = perft(&Game::new(start), d);
                let secs = t.elapsed().as_secs_f64();
                println!("depth {d}: {n} ({:.1} Mnodes/s)", n as f64 / secs / 1e6);
            }
            ExitCode::SUCCESS
        }
        Cmd::Difftest { probe, games, seed, workers, max_plies } => {
            let options = difftest::Options { probe, games, seed, workers, max_plies };
            if difftest::run(&options) { ExitCode::SUCCESS } else { ExitCode::FAILURE }
        }
        Cmd::Dump { games, seed, max_plies } => {
            dump::run(&dump::Options { games, seed, max_plies });
            ExitCode::SUCCESS
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perft_from_the_start() {
        let game = Game::new(Position::start());
        let counts: Vec<u64> = (1..=3).map(|d| perft(&game, d)).collect();
        assert_eq!(counts, [116, 6_788, 806_344]);
    }
}
