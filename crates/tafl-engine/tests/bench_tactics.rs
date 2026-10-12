//! Timing of the tactical analysis on recorded positions (one game's moves per line):
//! cargo test --release -p tafl-engine --test bench_tactics -- --ignored --nocapture
//! with TAFL_BENCH_GAMES pointing at the file.
use std::time::Instant;

use tafl_core::{Game, parse_move};
use tafl_engine::tactics::{Analysis, analyse};

#[test]
#[ignore]
fn bench_tactics() {
    let path = std::env::var("TAFL_BENCH_GAMES").expect("TAFL_BENCH_GAMES");
    let text = std::fs::read_to_string(path).expect("games file");
    let mut positions = Vec::new();
    for line in text.lines() {
        let mut game = Game::standard();
        for mv in line.split_whitespace() {
            if game.is_over() {
                break;
            }
            positions.push(game.clone());
            game.try_play(parse_move(mv).unwrap()).unwrap();
        }
    }
    let count = positions.len() as f64;
    let t = Instant::now();
    let generated: usize = positions.iter().map(|g| g.position().legal_moves().len()).sum();
    let movegen = t.elapsed().as_secs_f64() / count;
    let t = Instant::now();
    let (mut wins, mut losses, mut kept) = (0, 0, 0);
    for g in &positions {
        match analyse(g.position(), &|h| g.occurrences(h)) {
            Analysis::Win(_) => wins += 1,
            Analysis::Loss => losses += 1,
            Analysis::Moves(m) => kept += m.len(),
        }
    }
    let analysis = t.elapsed().as_secs_f64() / count;
    for side in [tafl_core::Side::Attackers, tafl_core::Side::Defenders] {
        let mut times: Vec<(f64, usize)> = positions
            .iter()
            .enumerate()
            .filter(|(_, g)| g.position().side == side)
            .map(|(i, g)| {
                let t = Instant::now();
                let _ = analyse(g.position(), &|h| g.occurrences(h));
                (t.elapsed().as_secs_f64() * 1e6, i)
            })
            .collect();
        let total: f64 = times.iter().map(|t| t.0).sum();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        let top: f64 = times[..times.len() / 100].iter().map(|t| t.0).sum();
        println!(
            "{side:?}: mean {:.1} us, slowest 1% take {:.0}% of the time, max {:.0} us at position {}",
            total / times.len() as f64,
            top / total * 100.0,
            times[0].0,
            times[0].1
        );
    }
    println!(
        "{count} positions: move generation {:.2} us, analysis {:.2} us; {wins} wins, {losses} losses, {kept} of {generated} moves kept",
        movegen * 1e6,
        analysis * 1e6
    );
}
