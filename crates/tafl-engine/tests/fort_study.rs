//! Where recorded exit-fort losses became unstoppable (one game's moves per line):
//! cargo test --release -p tafl-engine --test fort_study -- --ignored --nocapture
//! with TAFL_FORT_GAMES pointing at the file.
use tafl_core::{Game, Position, Reason, Side, parse_move};

fn fort_in_one(pos: &Position, occ: &impl Fn(u64) -> u32) -> bool {
    pos.legal_moves().iter().any(|&mv| {
        let mut next = *pos;
        next.play(mv);
        next.outcome(occ(next.hash) + 1).is_some_and(|o| o.reason == Reason::ExitFort)
    })
}

#[test]
#[ignore]
fn fort_study() {
    let path = std::env::var("TAFL_FORT_GAMES").expect("TAFL_FORT_GAMES");
    let text = std::fs::read_to_string(path).expect("games file");
    for (index, line) in text.lines().enumerate() {
        let moves: Vec<_> = line.split_whitespace().map(|m| parse_move(m).unwrap()).collect();
        let mut game = Game::standard();
        let mut report = Vec::new();
        for (ply, &mv) in moves.iter().enumerate() {
            let pos = *game.position();
            if pos.side == Side::Attackers {
                let occ = |h| game.occurrences(h);
                let mut threatened = pos;
                threatened.set_side(Side::Defenders);
                let threat = fort_in_one(&threatened, &occ);
                let legal = pos.legal_moves();
                let mut safe = 0;
                let mut played_safe = false;
                for &a in legal.iter() {
                    let mut next = pos;
                    next.play(a);
                    if next.outcome(occ(next.hash) + 1).is_some() {
                        continue;
                    }
                    if !fort_in_one(&next, &occ) {
                        safe += 1;
                        played_safe |= a == mv;
                    }
                }
                report.push(format!(
                    "{ply}:{}{}/{}{}",
                    if threat { "T" } else { "" },
                    safe,
                    legal.len(),
                    if played_safe { "" } else { "!" }
                ));
            }
            game.try_play(mv).unwrap();
        }
        let tail = report.len().saturating_sub(8);
        println!(
            "game {index} ({} plies, {:?}): {}",
            moves.len(),
            game.outcome().map(|o| o.reason),
            report[tail..].join(" ")
        );
    }
}
