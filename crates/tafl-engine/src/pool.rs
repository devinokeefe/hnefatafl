//! Many games searched in parallel, sharing one network batch.
//!
//! `collect` runs the selection phase of every game's search (in parallel) and writes
//! the queued leaves' input planes into one batch; the caller evaluates the batch and
//! hands the outputs back to `apply`. In self-play mode the pool plays its games to
//! the end on its own and emits training samples; in analysis mode the caller sets
//! up positions, starts searches and reads the results.

use rayon::prelude::*;
use tafl_core::rng::Rng;
use tafl_core::{Game, Move, Outcome, Position, Side};

use crate::encode::{CompactState, FEATURE_BYTES, HISTORY_MOVES, NUM_ACTIONS, action_index};
use crate::mcts::{Algorithm, Leaf, Search, SearchConfig, Step};

#[derive(Clone, Debug)]
pub struct PoolConfig {
    pub games: usize,
    pub search: SearchConfig,
    /// Self-play: simulations for recorded ("full") searches and for the others
    /// (playout-cap randomisation).
    pub full_simulations: u32,
    pub fast_simulations: u32,
    pub full_search_prob: f64,
    /// Leaves one search may queue per batch (virtual loss above 1).
    pub leaves_per_search: usize,
    /// Self-play games longer than this end with an unknown result.
    pub max_plies: usize,
    pub seed: u64,
    /// Resign when the searched value stays at or below `-resign_threshold` for
    /// `resign_moves` consecutive own moves (0 disables); a fraction of games never
    /// resigns, to measure how often resignation would have been wrong.
    pub resign_threshold: f32,
    pub resign_moves: u32,
    pub no_resign_fraction: f64,
    /// Self-play (true) or caller-driven analysis (false).
    pub autoplay: bool,
    /// Analysis with PUCT: a search for a game that continues the last searched one
    /// starts from the matching subtree of the last tree.
    pub reuse_tree: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig {
            games: 1,
            search: SearchConfig::default(),
            full_simulations: 128,
            fast_simulations: 32,
            full_search_prob: 0.25,
            leaves_per_search: 1,
            max_plies: 400,
            seed: 1,
            resign_threshold: 0.0,
            resign_moves: 3,
            no_resign_fraction: 0.1,
            autoplay: true,
            reuse_tree: false,
        }
    }
}

/// One training sample (fixed-size part; the policy target is stored separately).
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct SampleRecord {
    pub state: CompactState,
    /// Final result for the side to move: +1 win, -1 loss, 0 unknown.
    pub value: f32,
    /// Search value of the root for the side to move.
    pub root_q: f32,
    pub game_id: u32,
    pub ply: u16,
    /// Plies until the game ended (from this position).
    pub plies_left: u16,
    /// Action the opponent replied with, or `u16::MAX`.
    pub next_action: u16,
    pub policy_len: u16,
    /// 1 when the game's result is known (not cut off at the ply cap).
    pub result_known: u8,
    pub _pad: [u8; 11],
}

pub const SAMPLE_RECORD_BYTES: usize = std::mem::size_of::<SampleRecord>();

#[derive(Clone, Debug)]
pub struct GameSummary {
    pub id: u64,
    pub start_fen: String,
    pub moves: Vec<Move>,
    pub outcome: Option<Outcome>,
    pub resigned_by: Option<Side>,
    /// Whether resignation was disabled in this game (and who would have resigned).
    pub no_resign: bool,
    pub would_resign: Option<Side>,
}

struct PendingSample {
    state: CompactState,
    policy: Vec<(u16, f32)>,
    root_q: f32,
    side: Side,
    ply: usize,
}

pub struct FinishedGame {
    pub summary: GameSummary,
    pub samples: Vec<(SampleRecord, Vec<(u16, f32)>)>,
}

/// Result of a completed analysis search.
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub best: Move,
    pub value: f32,
    pub proof: i8,
    /// Visits of the root, including those kept from an earlier search of the game.
    pub simulations: u32,
    /// The simulations of this search alone.
    pub new_simulations: u32,
    pub pv: Vec<Move>,
    /// (move, visits, Q, prior) per root move.
    pub stats: Vec<(Move, u32, f32, f32)>,
    pub policy: Vec<(Move, f32)>,
}

struct Slot {
    game: Game,
    search: Option<Search>,
    full: bool,
    rng: Rng,
    leaves: Vec<Leaf>,
    samples: Vec<PendingSample>,
    game_id: u64,
    resign_streak: [u32; 2],
    no_resign: bool,
    would_resign: Option<Side>,
    finished: Vec<FinishedGame>,
    result: Option<SearchResult>,
    /// The last finished analysis tree and the game it searched (with `reuse_tree`).
    kept: Option<(Game, Search)>,
    start: Position,
}

impl Slot {
    fn new(seed: u64, start: Position) -> Slot {
        Slot {
            game: Game::new(start),
            search: None,
            full: false,
            rng: Rng::new(seed),
            leaves: Vec::new(),
            samples: Vec::new(),
            game_id: 0,
            resign_streak: [0; 2],
            no_resign: false,
            would_resign: None,
            finished: Vec::new(),
            result: None,
            kept: None,
            start,
        }
    }

    /// Selection for this slot's search; in self-play also finishes moves and games.
    fn select(&mut self, cfg: &PoolConfig, next_game_id: &std::sync::atomic::AtomicU64) {
        let mut budget = 10_000;
        while budget > 0 {
            budget -= 1;
            if cfg.autoplay && self.search.is_none() {
                if self.game_id == 0 {
                    self.begin_game(cfg, next_game_id);
                }
                self.full = self.rng.unit() < cfg.full_search_prob;
                let sims = if self.full { cfg.full_simulations } else { cfg.fast_simulations };
                self.search = Some(Search::new(&self.game, sims));
            }
            let Some(search) = self.search.as_mut() else { return };
            if search.is_complete() {
                if search.has_pending() {
                    return;
                }
                if cfg.autoplay {
                    self.finish_move(cfg, next_game_id);
                    continue;
                }
                self.result = Some(result_of(search, &cfg.search));
                let search = self.search.take();
                if cfg.reuse_tree {
                    self.kept = search.map(|s| (self.game.clone(), s));
                }
                return;
            }
            if self.leaves.len() >= cfg.leaves_per_search {
                return;
            }
            match search.select(&cfg.search, &mut self.leaves) {
                Step::Queued | Step::Resolved => {}
                Step::Collision => return,
            }
        }
    }

    fn begin_game(&mut self, cfg: &PoolConfig, next_game_id: &std::sync::atomic::AtomicU64) {
        self.game = Game::new(self.start);
        self.game_id = next_game_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.samples.clear();
        self.resign_streak = [0; 2];
        self.no_resign = cfg.resign_threshold <= 0.0 || self.rng.unit() < cfg.no_resign_fraction;
        self.would_resign = None;
    }

    fn finish_move(&mut self, cfg: &PoolConfig, next_game_id: &std::sync::atomic::AtomicU64) {
        let search = self.search.take().expect("finish_move without a search");
        let mv = search.best_move(&cfg.search);
        let side = self.game.position().side;
        let value = search.root_value();
        if self.full {
            let recent: Vec<Move> = self.game.moves().iter().rev().take(HISTORY_MOVES).copied().collect();
            let state =
                CompactState::new(self.game.position(), self.game.moves().len(), &recent, |h| self.game.occurrences(h));
            let policy =
                search.policy_target(&cfg.search).into_iter().map(|(m, p)| (action_index(m) as u16, p)).collect();
            self.samples.push(PendingSample { state, policy, root_q: value, side, ply: self.game.moves().len() });
        }
        let mut resigned = None;
        if cfg.resign_threshold > 0.0 {
            let streak = &mut self.resign_streak[side as usize];
            *streak = if value <= -cfg.resign_threshold { *streak + 1 } else { 0 };
            if *streak >= cfg.resign_moves.max(1) {
                if self.no_resign {
                    self.would_resign.get_or_insert(side);
                } else {
                    resigned = Some(side);
                }
            }
        }
        if resigned.is_none() {
            self.game.play_unchecked(mv);
        }
        if resigned.is_some() || self.game.is_over() || self.game.moves().len() >= cfg.max_plies {
            self.finish_game(resigned);
            self.begin_game(cfg, next_game_id);
        }
    }

    fn finish_game(&mut self, resigned_by: Option<Side>) {
        let outcome = self.game.outcome();
        let winner = outcome.map(|o| o.winner).or(resigned_by.map(Side::other));
        let total = self.game.moves().len();
        let samples = self
            .samples
            .drain(..)
            .map(|s| {
                let next_action = self.game.moves().get(s.ply + 1).map_or(u16::MAX, |m| action_index(*m) as u16);
                let record = SampleRecord {
                    state: s.state,
                    value: match winner {
                        Some(w) if w == s.side => 1.0,
                        Some(_) => -1.0,
                        None => 0.0,
                    },
                    root_q: s.root_q,
                    game_id: self.game_id as u32,
                    ply: s.ply as u16,
                    plies_left: (total - s.ply) as u16,
                    next_action,
                    policy_len: s.policy.len() as u16,
                    result_known: u8::from(winner.is_some()),
                    _pad: [0; 11],
                };
                (record, s.policy)
            })
            .collect();
        self.finished.push(FinishedGame {
            summary: GameSummary {
                id: self.game_id,
                start_fen: self.start.fen(),
                moves: self.game.moves().to_vec(),
                outcome,
                resigned_by,
                no_resign: self.no_resign,
                would_resign: self.would_resign,
            },
            samples,
        });
    }
}

fn result_of(search: &Search, cfg: &SearchConfig) -> SearchResult {
    SearchResult {
        best: search.best_move(cfg),
        value: search.root_value(),
        proof: search.root_proof(),
        simulations: search.simulations_done(),
        new_simulations: search.new_simulations(),
        pv: search.principal_variation(12),
        stats: search.root_stats(),
        policy: search.policy_target(cfg),
    }
}

pub struct Pool {
    cfg: PoolConfig,
    slots: Vec<Slot>,
    /// Batch rows per slot from the last `collect`: (first row, count).
    rows: Vec<(usize, usize)>,
    next_game_id: std::sync::atomic::AtomicU64,
    finished: Vec<FinishedGame>,
}

impl Pool {
    pub fn new(cfg: PoolConfig) -> Pool {
        let mut seeder = Rng::new(cfg.seed);
        let slots = (0..cfg.games).map(|_| Slot::new(seeder.next_u64(), Position::start())).collect();
        Pool { slots, rows: Vec::new(), next_game_id: std::sync::atomic::AtomicU64::new(1), finished: Vec::new(), cfg }
    }

    pub fn config(&self) -> &PoolConfig {
        &self.cfg
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Maximum rows one `collect` can produce.
    pub fn max_batch(&self) -> usize {
        self.slots.len() * self.cfg.leaves_per_search
    }

    /// Runs selection everywhere and writes the queued leaves' planes into `planes`
    /// (`FEATURE_BYTES` per row). Returns the number of rows.
    pub fn collect(&mut self, planes: &mut [u8]) -> usize {
        let cfg = &self.cfg;
        let ids = &self.next_game_id;
        self.slots.par_iter_mut().for_each(|slot| {
            if slot.leaves.is_empty() {
                slot.select(cfg, ids);
            }
        });
        self.rows.clear();
        let mut row = 0;
        for slot in &mut self.slots {
            self.rows.push((row, slot.leaves.len()));
            row += slot.leaves.len();
            self.finished.append(&mut slot.finished);
        }
        assert!(planes.len() >= row * FEATURE_BYTES, "plane buffer too small");
        let mut chunks: Vec<&mut [u8]> = planes[..row * FEATURE_BYTES].chunks_mut(FEATURE_BYTES).collect();
        let mut index = 0;
        for slot in &self.slots {
            for leaf in &slot.leaves {
                leaf.state.write_planes(0, chunks[index]);
                index += 1;
            }
        }
        let _ = &mut chunks;
        row
    }

    /// Feeds the evaluation of the last batch: `policy` holds `NUM_ACTIONS` logits per
    /// row, `value` the value for the side to move per row.
    pub fn apply(&mut self, policy: &[f32], value: &[f32]) {
        let cfg = &self.cfg;
        let rows = &self.rows;
        self.slots.par_iter_mut().enumerate().for_each(|(i, slot)| {
            let (first, count) = rows.get(i).copied().unwrap_or((0, 0));
            debug_assert_eq!(count, slot.leaves.len());
            if count == 0 {
                return;
            }
            let leaves: Vec<Leaf> = std::mem::take(&mut slot.leaves);
            let search = slot.search.as_mut().expect("leaves without a search");
            for (k, leaf) in leaves.into_iter().enumerate() {
                let r = first + k;
                search.apply(
                    leaf,
                    &policy[r * NUM_ACTIONS..(r + 1) * NUM_ACTIONS],
                    value[r],
                    &cfg.search,
                    &mut slot.rng,
                );
            }
        });
        self.rows.clear();
    }

    /// Takes the games finished so far.
    pub fn take_finished(&mut self) -> Vec<FinishedGame> {
        std::mem::take(&mut self.finished)
    }

    // ----- analysis mode -----

    pub fn set_game(&mut self, slot: usize, game: Game) {
        let s = &mut self.slots[slot];
        s.game = game;
        s.search = None;
        s.leaves.clear();
        s.result = None;
    }

    pub fn game(&self, slot: usize) -> &Game {
        &self.slots[slot].game
    }

    pub fn play(&mut self, slot: usize, mv: Move) -> Result<(), String> {
        let s = &mut self.slots[slot];
        s.search = None;
        s.leaves.clear();
        s.result = None;
        s.game.try_play(mv)
    }

    pub fn start_search(&mut self, slot: usize, simulations: u32) {
        let reuse = self.cfg.reuse_tree && self.cfg.search.algorithm == Algorithm::Puct;
        let s = &mut self.slots[slot];
        s.leaves.clear();
        s.result = None;
        let reused = s.kept.take().filter(|_| reuse).and_then(|(old, tree)| {
            let (before, now) = (old.moves(), s.game.moves());
            let continues = old.start().hash == s.game.start().hash && now.starts_with(before);
            continues.then(|| tree.reroot(&now[before.len()..], &s.game, simulations)).flatten()
        });
        s.search = Some(reused.unwrap_or_else(|| Search::new(&s.game, simulations)));
    }

    /// Stops a running analysis search early; its result is the best move so far.
    pub fn stop_search(&mut self, slot: usize) {
        if let Some(search) = self.slots[slot].search.as_mut() {
            search.stop();
        }
    }

    pub fn is_thinking(&self, slot: usize) -> bool {
        self.slots[slot].search.is_some()
    }

    pub fn result(&self, slot: usize) -> Option<&SearchResult> {
        self.slots[slot].result.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_round(pool: &mut Pool) -> usize {
        let mut planes = vec![0u8; pool.max_batch() * FEATURE_BYTES];
        let n = pool.collect(&mut planes);
        let policy = vec![0.0f32; n * NUM_ACTIONS];
        let value = vec![0.0f32; n];
        pool.apply(&policy, &value);
        n
    }

    #[test]
    fn self_play_produces_consistent_samples() {
        let cfg = PoolConfig {
            games: 4,
            full_simulations: 16,
            fast_simulations: 4,
            full_search_prob: 0.5,
            max_plies: 60,
            ..PoolConfig::default()
        };
        let mut pool = Pool::new(cfg);
        let mut games = Vec::new();
        for _ in 0..20_000 {
            uniform_round(&mut pool);
            games.extend(pool.take_finished());
            if games.len() >= 6 {
                break;
            }
        }
        assert!(games.len() >= 6);
        for g in &games {
            let mut replay = Game::standard();
            for &m in &g.summary.moves {
                replay.try_play(m).unwrap();
            }
            assert_eq!(replay.outcome(), g.summary.outcome);
            for (record, policy) in &g.samples {
                let ply = record.ply as usize;
                let mut at = Game::standard();
                for &m in &g.summary.moves[..ply] {
                    at.play_unchecked(m);
                }
                assert_eq!(record.state.position().fen(), at.position().fen());
                let total: f32 = policy.iter().map(|p| p.1).sum();
                assert!((total - 1.0).abs() < 1e-3);
                assert_eq!(record.result_known == 1, g.summary.outcome.is_some());
                if let Some(o) = g.summary.outcome {
                    let expected = if o.winner == at.position().side { 1.0 } else { -1.0 };
                    assert_eq!(record.value, expected);
                }
            }
        }
    }

    #[test]
    fn analysis_mode_with_virtual_loss() {
        let cfg = PoolConfig { games: 2, autoplay: false, leaves_per_search: 8, ..PoolConfig::default() };
        let mut pool = Pool::new(cfg);
        pool.start_search(0, 200);
        pool.start_search(1, 50);
        let mut rounds = 0;
        while pool.is_thinking(0) || pool.is_thinking(1) {
            uniform_round(&mut pool);
            rounds += 1;
            assert!(rounds < 1000);
        }
        assert!(rounds < 200 / 2, "virtual loss should batch several leaves per round ({rounds})");
        let result = pool.result(0).unwrap();
        assert!(result.simulations >= 200);
        assert!(Game::standard().position().is_legal(result.best));
    }

    #[test]
    fn analysis_reuses_the_tree_of_the_game_it_continues() {
        let search = SearchConfig { algorithm: Algorithm::Puct, ..SearchConfig::default() };
        let cfg =
            PoolConfig { autoplay: false, reuse_tree: true, leaves_per_search: 4, search, ..PoolConfig::default() };
        let mut pool = Pool::new(cfg);
        let think = |pool: &mut Pool, sims: u32| {
            pool.start_search(0, sims);
            while pool.is_thinking(0) {
                uniform_round(pool);
            }
            pool.result(0).unwrap().clone()
        };
        let first = think(&mut pool, 800);
        // Searching the same position again continues the whole tree.
        let again = think(&mut pool, 100);
        assert!(again.simulations >= 900);
        assert!((100..104).contains(&again.new_simulations));
        // After the expected line the search starts from that subtree.
        let mut game = Game::standard();
        game.try_play(first.pv[0]).unwrap();
        game.try_play(first.pv[1]).unwrap();
        pool.set_game(0, game);
        let second = think(&mut pool, 100);
        assert!(second.simulations >= 100);
        // An unrelated game starts afresh.
        pool.set_game(0, Game::standard());
        assert!(think(&mut pool, 100).simulations < 100 + 4);
    }
}
