//! Differential test of our rules against TaflZero's own implementation.
//!
//! TaflZero's rules are reached through its research probe (a JSON-lines program
//! linked against the production engine library): for a start FEN and a move list it
//! reports the resulting FEN, legal moves, repetition count and terminal result. We
//! play random games, biased towards captures, edge play, king walks and ring
//! building so that shieldwalls, forts and encirclements actually occur, and compare
//! every ply. `tafl dump` writes games from the same generator.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

use serde_json::{Value, json};
use tafl_core::geometry::{CORNERS, KING_RING, RESTRICTED, THRONE, is_edge};
use tafl_core::rng::Rng;
use tafl_core::symmetry::{NUM_SYMMETRIES, transform_position};
use tafl_core::{Bb, Game, Move, NUM_SQUARES, Piece, Position, SIZE, Side, Sq, square};

pub struct Options {
    pub probe: PathBuf,
    pub games: usize,
    pub seed: u64,
    pub workers: usize,
    pub max_plies: usize,
}

struct Probe {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Probe {
    fn spawn(path: &PathBuf) -> Probe {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap_or_else(|e| panic!("cannot start probe {}: {e}", path.display()));
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Probe { child, stdin, stdout }
    }

    fn query(&mut self, fen: &str, moves: &[Move]) -> Value {
        let moves: Vec<String> = moves.iter().map(|m| m.to_string()).collect();
        writeln!(self.stdin, "{}", json!({"fen": fen, "moves": moves})).unwrap();
        self.stdin.flush().unwrap();
        let mut line = String::new();
        self.stdout.read_line(&mut line).unwrap();
        serde_json::from_str(&line).unwrap_or_else(|e| panic!("bad probe output ({e}): {line}"))
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// How a test game starts and chooses its moves.
#[derive(Clone, Copy, Debug)]
pub enum Style {
    Uniform,
    Tactical,
    Setup,
    /// Few defenders against many attackers: encirclements.
    Siege,
    /// Undo the side's own previous move when possible: repetitions.
    Shuffle,
    /// Crowded edges: shieldwalls.
    Walls,
    /// Defenders inside a gapped ring of attackers: encirclements.
    Ring,
    /// Roofed runs along one edge, ready to be bracketed: shieldwalls, single-piece
    /// runs included. Used by `tafl dump` (not in `ALL`, so difftest runs are unchanged).
    Edge,
}

impl Style {
    /// Game `index` of a run is played in style `ALL[index % ALL.len()]`.
    pub const ALL: [Style; 7] =
        [Style::Uniform, Style::Tactical, Style::Setup, Style::Siege, Style::Shuffle, Style::Walls, Style::Ring];
}

/// The random generator of game `index` in a run with this seed.
pub fn game_rng(seed: u64, index: usize) -> Rng {
    Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(index as u64))
}

/// The starting position of a game in this style.
pub fn start_position(style: Style, rng: &mut Rng) -> Position {
    match style {
        Style::Setup => random_setup(rng, 12),
        Style::Siege => random_setup(rng, 3),
        Style::Walls => walls_setup(rng),
        Style::Ring => ring_setup(rng),
        Style::Edge => edge_setup(rng),
        _ => Position::start(),
    }
}

/// The next move of a running game in this style.
pub fn choose_move(style: Style, game: &Game, rng: &mut Rng) -> Move {
    let pos = game.position();
    let plies = game.moves().len();
    let undo = (plies >= 2).then(|| game.moves()[plies - 2]).map(|m| Move::new(m.to, m.from));
    match style {
        Style::Shuffle if undo.is_some_and(|m| pos.is_legal(m)) && rng.below(10) < 8 => undo.unwrap(),
        Style::Uniform | Style::Shuffle => {
            let moves = pos.legal_moves();
            moves[rng.below(moves.len())]
        }
        Style::Edge => {
            // Mostly take a capture of several pieces when there is one: shieldwalls here.
            let takes = |m: Move| {
                let mut next = *pos;
                next.play(m).count()
            };
            let big: Vec<Move> = pos.legal_moves().iter().copied().filter(|&m| takes(m) >= 2).collect();
            if !big.is_empty() && rng.below(4) != 0 { big[rng.below(big.len())] } else { tactical_move(game, rng) }
        }
        Style::Tactical | Style::Setup | Style::Siege | Style::Walls | Style::Ring => tactical_move(game, rng),
    }
}

/// Whether a move's captures include a shieldwall's: a piece not beside the arrival
/// square (sandwiches and the king's capture take only neighbours).
pub fn is_shieldwall(mv: Move, captured: Bb, after: &Position) -> bool {
    (captured & !Bb::from_sq(mv.to).neighbors() & !after.king_bb()).any()
}

/// Pieces of random sides on most edge and second-line squares.
fn walls_setup(rng: &mut Rng) -> Position {
    loop {
        let mut pos = Position::empty();
        let king = loop {
            let sq = rng.below(NUM_SQUARES) as Sq;
            if !CORNERS.has(sq) {
                break sq;
            }
        };
        pos.put(king, Piece::King);
        for sq in 0..NUM_SQUARES as Sq {
            let (r, c) = (tafl_core::geometry::row_of(sq), tafl_core::geometry::col_of(sq));
            let near_edge = r <= 1 || r >= 9 || c <= 1 || c >= 9;
            if sq == king || RESTRICTED.has(sq) || !near_edge || rng.below(10) < 4 {
                continue;
            }
            pos.put(sq, if rng.below(2) == 0 { Piece::Attacker } else { Piece::Defender });
        }
        pos.set_side(if rng.below(2) == 0 { Side::Attackers } else { Side::Defenders });
        if pos.outcome(0).is_none() && pos.attackers.any() {
            return pos;
        }
    }
}

/// King and defenders in a box whose attacker border has a few gaps.
fn ring_setup(rng: &mut Rng) -> Position {
    loop {
        let mut pos = Position::empty();
        let (r0, c0) = (1 + rng.below(4), 1 + rng.below(4));
        let (r1, c1) = (r0 + 3 + rng.below(9 - r0 - 2).min(4), c0 + 3 + rng.below(9 - c0 - 2).min(4));
        let mut inside = Vec::new();
        for r in r0..=r1.min(9) {
            for c in c0..=c1.min(9) {
                let sq = tafl_core::square(r, c);
                let border = r == r0 || r == r1.min(9) || c == c0 || c == c1.min(9);
                if RESTRICTED.has(sq) {
                    continue;
                }
                if border {
                    if rng.below(10) < 8 {
                        pos.put(sq, Piece::Attacker);
                    }
                } else {
                    inside.push(sq);
                }
            }
        }
        if inside.len() < 2 {
            continue;
        }
        let king = inside.swap_remove(rng.below(inside.len()));
        pos.put(king, Piece::King);
        for _ in 0..rng.below(inside.len().min(5) + 1) {
            let sq = inside.swap_remove(rng.below(inside.len()));
            pos.put(sq, Piece::Defender);
        }
        for _ in 0..rng.below(6) {
            let sq = rng.below(NUM_SQUARES) as Sq;
            if pos.piece_at(sq).is_none() && !RESTRICTED.has(sq) {
                pos.put(sq, Piece::Attacker);
            }
        }
        pos.set_side(Side::Attackers);
        if pos.outcome(0).is_none() {
            return pos;
        }
    }
}

/// Rank 1 lined with soldiers, most enemies of the side to move roofed by its pieces, and
/// one of its pieces ready to drop onto each empty edge square; turned onto a random edge.
fn edge_setup(rng: &mut Rng) -> Position {
    loop {
        let mover = if rng.below(2) == 0 { Side::Attackers } else { Side::Defenders };
        let (friend, enemy) = if mover == Side::Attackers {
            (Piece::Attacker, Piece::Defender)
        } else {
            (Piece::Defender, Piece::Attacker)
        };
        let mut pos = Position::empty();
        for col in 1..SIZE - 1 {
            match rng.below(10) {
                0..5 => {
                    pos.put(square(0, col), enemy);
                    if rng.below(10) < 9 {
                        pos.put(square(1, col), friend);
                    }
                }
                5..8 => pos.put(square(0, col), friend),
                _ => {}
            }
        }
        for col in 1..SIZE - 1 {
            if pos.piece_at(square(0, col)).is_none() && pos.piece_at(square(1, col)).is_none() {
                pos.put(square(2 + rng.below(3), col), friend);
            }
        }
        // The king among the edge defenders now and then, otherwise anywhere free.
        let edge_defenders: Vec<Sq> = (1..SIZE - 1).map(|c| square(0, c)).filter(|&s| pos.defenders.has(s)).collect();
        let king = if !edge_defenders.is_empty() && rng.below(3) == 0 {
            let sq = edge_defenders[rng.below(edge_defenders.len())];
            pos.remove(sq);
            sq
        } else {
            loop {
                let sq = rng.below(NUM_SQUARES) as Sq;
                if pos.piece_at(sq).is_none() && !RESTRICTED.has(sq) {
                    break sq;
                }
            }
        };
        pos.put(king, Piece::King);
        for _ in 0..rng.below(7) {
            let sq = rng.below(NUM_SQUARES) as Sq;
            if pos.piece_at(sq).is_none() && !RESTRICTED.has(sq) {
                pos.put(sq, if rng.below(2) == 0 { Piece::Attacker } else { Piece::Defender });
            }
        }
        let mut pos = transform_position(rng.below(NUM_SYMMETRIES), &pos);
        pos.set_side(mover);
        if pos.outcome(0).is_none() && pos.attackers.any() {
            return pos;
        }
    }
}

/// A random position with random material (soldiers never on corners or throne).
fn random_setup(rng: &mut Rng, max_defenders: usize) -> Position {
    loop {
        let mut pos = Position::empty();
        let mut free: Vec<Sq> = (0..NUM_SQUARES as Sq).filter(|&s| !RESTRICTED.has(s)).collect();
        let mut take = |rng: &mut Rng| free.swap_remove(rng.below(free.len()));
        let king_sq = if rng.below(4) == 0 { THRONE } else { take(rng) };
        pos.put(king_sq, Piece::King);
        for _ in 0..1 + rng.below(24) {
            pos.put(take(rng), Piece::Attacker);
        }
        for _ in 0..rng.below(max_defenders + 1) {
            pos.put(take(rng), Piece::Defender);
        }
        pos.set_side(if rng.below(2) == 0 { Side::Attackers } else { Side::Defenders });
        if pos.outcome(0).is_none() {
            return pos;
        }
    }
}

/// A move chosen with weights that favour rule-relevant events.
fn tactical_move(game: &Game, rng: &mut Rng) -> Move {
    let pos = game.position();
    let moves = pos.legal_moves();
    let mut weights = Vec::with_capacity(moves.len());
    for &mv in moves.iter() {
        let mut next = *pos;
        let captured = next.play(mv);
        let mut w = 1.0;
        if captured.any() {
            w += if captured.count() >= 2 { 60.0 } else { 20.0 };
        }
        if next.outcome(1).is_some_and(|o| o.reason != tafl_core::Reason::KingEscaped) {
            w += 60.0;
        }
        if is_edge(mv.to) {
            w += 3.0;
        }
        match pos.side {
            Side::Attackers => {
                if (KING_RING[mv.to as usize] & next.attackers).count() >= 2 {
                    w += 6.0;
                }
            }
            Side::Defenders => {
                if mv.from == pos.king {
                    w += if is_edge(mv.to) { 15.0 } else { 5.0 };
                    if CORNERS.has(mv.to) {
                        w = 2.0; // Do not end every game by escaping at once.
                    }
                } else if next.has_king() && KING_RING[next.king as usize].has(mv.to) {
                    w += 8.0;
                }
            }
        }
        weights.push(w);
    }
    let total: f64 = weights.iter().sum();
    let mut x = rng.unit() * total;
    for (i, w) in weights.iter().enumerate() {
        x -= w;
        if x <= 0.0 {
            return moves[i];
        }
    }
    moves[moves.len() - 1]
}

#[derive(Default)]
struct Tally {
    positions: u64,
    games: u64,
    mismatches: Vec<String>,
    endings: BTreeMap<String, u64>,
    shieldwalls: u64,
}

fn compare(game: &Game, start_fen: &str, probe: &Value) -> Option<String> {
    let pos = game.position();
    let mut problems = Vec::new();
    if probe["fen"].as_str() != Some(pos.fen().as_str()) {
        problems.push(format!("fen ours {} theirs {}", pos.fen(), probe["fen"]));
    }
    let mut ours: Vec<String> = pos.legal_moves().iter().map(|m| m.to_string()).collect();
    let mut theirs: Vec<String> =
        probe["moves"].as_array().unwrap().iter().map(|m| m["move"].as_str().unwrap().to_string()).collect();
    ours.sort();
    theirs.sort();
    if ours != theirs {
        problems.push(format!("moves differ ({} vs {})", ours.len(), theirs.len()));
    }
    let reps = game.occurrences(pos.hash) as u64;
    if probe["rep_count"].as_u64() != Some(reps) {
        problems.push(format!("rep count ours {reps} theirs {}", probe["rep_count"]));
    }
    let (winner, code) = match game.outcome() {
        Some(o) => {
            let w = if o.winner == Side::Attackers { "ATTACKERS" } else { "DEFENDERS" };
            (Value::from(w), Value::from(o.reason.taflzero_code(o.winner.other())))
        }
        None => (Value::Null, Value::Null),
    };
    if probe["terminal"] != winner || probe["reason"] != code {
        problems.push(format!("terminal ours {winner}/{code} theirs {}/{}", probe["terminal"], probe["reason"]));
    }
    if problems.is_empty() {
        return None;
    }
    let moves: Vec<String> = game.moves().iter().map(|m| m.to_string()).collect();
    Some(format!("{}\n  start {start_fen}\n  moves {}", problems.join("; "), moves.join(" ")))
}

fn run_game(probe: &mut Probe, rng: &mut Rng, style: Style, max_plies: usize, tally: &mut Tally) {
    let start = start_position(style, rng);
    let start_fen = start.fen();
    let mut game = Game::new(start);
    loop {
        let reply = probe.query(&start_fen, game.moves());
        tally.positions += 1;
        if let Some(problem) = compare(&game, &start_fen, &reply) {
            tally.mismatches.push(problem);
            return;
        }
        if let Some(outcome) = game.outcome() {
            *tally.endings.entry(format!("{:?}", outcome.reason)).or_default() += 1;
            break;
        }
        if game.moves().len() >= max_plies {
            *tally.endings.entry("ply cap".into()).or_default() += 1;
            break;
        }
        let mv = choose_move(style, &game, rng);
        let mut after = *game.position();
        let captured = after.play(mv);
        if is_shieldwall(mv, captured, &after) {
            tally.shieldwalls += 1;
        }
        game.play_unchecked(mv);
    }
    tally.games += 1;
}

pub fn run(options: &Options) -> bool {
    let total = Mutex::new(Tally::default());
    let next_game = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..options.workers {
            scope.spawn(|| {
                let mut probe = Probe::spawn(&options.probe);
                let mut tally = Tally::default();
                loop {
                    let index = next_game.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if index >= options.games {
                        break;
                    }
                    let mut rng = game_rng(options.seed, index);
                    let style = Style::ALL[index % Style::ALL.len()];
                    run_game(&mut probe, &mut rng, style, options.max_plies, &mut tally);
                }
                let mut total = total.lock().unwrap();
                total.positions += tally.positions;
                total.games += tally.games;
                total.mismatches.extend(tally.mismatches);
                total.shieldwalls += tally.shieldwalls;
                for (k, v) in tally.endings {
                    *total.endings.entry(k).or_default() += v;
                }
            });
        }
    });
    let total = total.into_inner().unwrap();
    println!("compared {} positions in {} games", total.positions, total.games + total.mismatches.len() as u64);
    println!("game endings: {:?}", total.endings);
    println!("moves with shieldwall captures: {}", total.shieldwalls);
    for problem in total.mismatches.iter().take(10) {
        println!("MISMATCH {problem}");
    }
    println!("{} mismatching games", total.mismatches.len());
    total.mismatches.is_empty()
}
