//! Monte Carlo tree search with batched network evaluation and a proof solver.
//!
//! Two selection schemes are available: Gumbel AlphaZero (sequential halving at the
//! root, deterministic "improved policy" selection below it; Danihelka et al. 2022),
//! which gives a policy improvement even with few simulations and is used for
//! self-play, and PUCT with first-play urgency. Leaves are evaluated in batches by
//! the caller; a search hands out leaves with `select` and receives the network
//! outputs through `apply`.
//!
//! Terminal positions and the exact one-move tactics of [`crate::tactics`] give proven
//! results, which are propagated upwards (a node is won if some child is lost for the
//! opponent, lost if all children are won for the opponent). Node statistics are
//! path-dependent because repetition counts depend on the path: the tree is a tree,
//! not a graph.

use tafl_core::rng::Rng;
use tafl_core::{Game, Move, OccurrenceMap, Position, Side};

use crate::encode::{CompactState, HISTORY_MOVES, NUM_ACTIONS, action_index};
use crate::tactics::{Analysis, analyse, threatens_fort};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Algorithm {
    Gumbel,
    Puct,
}

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub algorithm: Algorithm,
    /// Gumbel: actions sampled at the root for sequential halving.
    pub considered_actions: usize,
    /// Gumbel: sample root noise (off for deterministic play).
    pub gumbel_noise: bool,
    /// Gumbel: sigma(q) = (maxvisit_init + max visits) * value_scale * q.
    pub value_scale: f32,
    pub maxvisit_init: f32,
    pub c_puct: f32,
    pub fpu_reduction: f32,
    /// Softmax temperature applied to the network's policy logits.
    pub policy_temperature: f32,
    /// Use exact tactics at expansion.
    pub tactics: bool,
    /// Gumbel: give every defender root move that threatens to complete an exit fort
    /// one of the first visits of sequential halving, so that self-play tries fort
    /// building even while the policy does not yet propose it. A threat the attackers
    /// cannot meet is then proven at once by the tactics.
    pub fort_threats: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            algorithm: Algorithm::Gumbel,
            considered_actions: 16,
            gumbel_noise: true,
            value_scale: 0.1,
            maxvisit_init: 50.0,
            c_puct: 1.5,
            fpu_reduction: 0.25,
            policy_temperature: 1.0,
            tactics: true,
            fort_threats: false,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    New,
    Pending,
    Expanded,
    Terminal,
}

/// Statistics of a position in the tree. Values are from the point of view of the
/// player who moved into the node; `raw_value` is for the side to move in it.
#[derive(Clone, Copy, Debug)]
struct Node {
    visits: u32,
    virtual_loss: u32,
    value_sum: f32,
    raw_value: f32,
    edge_start: u32,
    num_edges: u32,
    state: State,
    /// +1: the player who moved into this node wins with best play; -1: loses.
    proof: i8,
}

impl Node {
    fn new() -> Node {
        Node {
            visits: 0,
            virtual_loss: 0,
            value_sum: 0.0,
            raw_value: 0.0,
            edge_start: 0,
            num_edges: 0,
            state: State::New,
            proof: 0,
        }
    }
}

/// A move out of a node; `child` is 0 until the move is first visited.
#[derive(Clone, Copy, Debug)]
struct Edge {
    mv: Move,
    logit: f32,
    prior: f32,
    child: u32,
}

/// Repetition counts and recent moves of the game before the search root.
#[derive(Clone, Debug, Default)]
pub struct RootContext {
    counts: OccurrenceMap,
    /// Most recent first, at most `HISTORY_MOVES`.
    recent: Vec<Move>,
    ply: usize,
}

impl RootContext {
    pub fn from_game(game: &Game) -> RootContext {
        RootContext {
            counts: game.occurrence_map().clone(),
            recent: game.moves().iter().rev().take(HISTORY_MOVES).copied().collect(),
            ply: game.moves().len(),
        }
    }
}

/// A leaf waiting for a network evaluation.
pub struct Leaf {
    node: u32,
    path: Vec<u32>,
    moves: Vec<Move>,
    pub state: CompactState,
}

pub enum Step {
    /// A leaf was queued for evaluation.
    Queued,
    /// A simulation finished without the network (terminal or proven node).
    Resolved,
    /// The descent hit a leaf already awaiting evaluation.
    Collision,
}

pub struct Search {
    root: Position,
    ctx: RootContext,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    /// Gumbel noise per root edge and the sequential-halving visit schedule.
    gumbel: Vec<f32>,
    schedule: Vec<u32>,
    /// Root edges given a first visit ahead of the others (see ort_threats).
    forced: Vec<bool>,
    target: u32,
    /// Root visits taken over from an earlier search (see `reroot`).
    kept: u32,
}

/// Visit counts at which each simulation of sequential halving is spent (as in mctx's
/// `get_sequence_of_considered_visits`).
fn halving_schedule(considered: usize, simulations: u32) -> Vec<u32> {
    if considered <= 1 {
        return (0..simulations).collect();
    }
    let log2 = (considered as f64).log2().ceil() as u32;
    let mut sequence = Vec::with_capacity(simulations as usize);
    let mut visits = vec![0u32; considered];
    let mut live = considered;
    while (sequence.len() as u32) < simulations {
        let extra = (simulations / (log2 * live as u32)).max(1);
        for _ in 0..extra {
            sequence.extend_from_slice(&visits[..live]);
            for v in visits.iter_mut().take(live) {
                *v += 1;
            }
        }
        live = (live / 2).max(2);
    }
    sequence.truncate(simulations as usize);
    sequence
}

impl Search {
    pub fn new(game: &Game, simulations: u32) -> Search {
        Search::with_context(*game.position(), RootContext::from_game(game), simulations)
    }

    pub fn with_context(root: Position, ctx: RootContext, simulations: u32) -> Search {
        Search {
            root,
            ctx,
            nodes: vec![Node::new()],
            edges: Vec::new(),
            gumbel: Vec::new(),
            schedule: Vec::new(),
            forced: Vec::new(),
            target: simulations.max(1),
            kept: 0,
        }
    }

    /// The subtree reached by `moves` from this search's root, as a new search for
    /// `game` (whose position those moves lead to) that keeps its statistics and runs
    /// `simulations` more; None if that position was not expanded. Values and proofs
    /// stay exact: repetition counts along the old path become the new game history.
    /// For PUCT only (the Gumbel root schedule assumes a fresh root).
    pub fn reroot(&self, moves: &[Move], game: &Game, simulations: u32) -> Option<Search> {
        let mut node = 0usize;
        for &mv in moves {
            if self.nodes[node].state != State::Expanded {
                return None;
            }
            let edge = self.edges_of(node).iter().find(|e| e.mv == mv)?;
            if edge.child == 0 {
                return None;
            }
            node = edge.child as usize;
        }
        if self.nodes[node].state != State::Expanded {
            return None;
        }
        debug_assert_eq!(self.nodes[node].virtual_loss, 0);
        let mut out = Search::with_context(*game.position(), RootContext::from_game(game), 1);
        out.nodes[0] = self.nodes[node];
        let mut stack = vec![(node, 0usize)];
        while let Some((old, new)) = stack.pop() {
            if self.nodes[old].state != State::Expanded {
                continue;
            }
            out.nodes[new].edge_start = out.edges.len() as u32;
            for e in self.edges_of(old) {
                let mut copy = *e;
                if e.child != 0 {
                    let mut child = self.nodes[e.child as usize];
                    if child.state == State::Pending {
                        child.state = State::New;
                    }
                    child.virtual_loss = 0;
                    copy.child = out.nodes.len() as u32;
                    stack.push((e.child as usize, out.nodes.len()));
                    out.nodes.push(child);
                }
                out.edges.push(copy);
            }
        }
        out.kept = out.nodes[0].visits;
        out.target = out.kept + simulations.max(1);
        Some(out)
    }

    pub fn root_position(&self) -> &Position {
        &self.root
    }

    pub fn simulations_done(&self) -> u32 {
        self.nodes[0].visits
    }

    /// The simulations this search ran itself, without those kept from an earlier one.
    pub fn new_simulations(&self) -> u32 {
        self.nodes[0].visits - self.kept
    }

    pub fn target(&self) -> u32 {
        self.target
    }

    /// Ends the search early (for time limits): it completes once the leaves already
    /// handed out have been applied.
    pub fn stop(&mut self) {
        self.target = self.target.min(self.nodes[0].visits.max(1));
    }

    /// Done when the budget is spent or the root's result is proven.
    pub fn is_complete(&self) -> bool {
        let root = &self.nodes[0];
        root.state == State::Expanded && (root.visits >= self.target || root.proof != 0)
    }

    pub fn has_pending(&self) -> bool {
        self.nodes[0].virtual_loss > 0
    }

    fn occurrences(&self, hash: u64, path_hashes: &[u64]) -> u32 {
        self.ctx.counts.get(&hash).copied().unwrap_or(0) + path_hashes.iter().filter(|&&h| h == hash).count() as u32
    }

    /// One descent from the root: queues a leaf in `out`, or resolves a simulation.
    pub fn select(&mut self, cfg: &SearchConfig, out: &mut Vec<Leaf>) -> Step {
        let mut path = vec![0u32];
        let mut pos = self.root;
        let mut path_moves: Vec<Move> = Vec::new();
        let mut path_hashes: Vec<u64> = Vec::new();
        let mut node = 0usize;
        loop {
            match self.nodes[node].state {
                State::Terminal => {
                    let v = self.nodes[node].raw_value;
                    self.backup(&path, v);
                    return Step::Resolved;
                }
                State::Pending => return Step::Collision,
                State::Expanded => {
                    let proof = self.nodes[node].proof;
                    if proof != 0 {
                        self.backup(&path, -(proof as f32));
                        return Step::Resolved;
                    }
                    let e = self.pick_edge(node, cfg);
                    let mv = self.edges[e].mv;
                    let mover = pos.side;
                    pos.play(mv);
                    path_moves.push(mv);
                    path_hashes.push(pos.hash);
                    if self.edges[e].child == 0 {
                        let child = self.nodes.len() as u32;
                        self.edges[e].child = child;
                        let mut fresh = Node::new();
                        if let Some(outcome) = pos.outcome(self.occurrences(pos.hash, &path_hashes)) {
                            fresh.state = State::Terminal;
                            let mover_wins = outcome.winner == mover;
                            fresh.proof = if mover_wins { 1 } else { -1 };
                            fresh.raw_value = if mover_wins { -1.0 } else { 1.0 };
                        }
                        self.nodes.push(fresh);
                        if fresh.state == State::Terminal {
                            path.push(child);
                            self.propagate_proof(&path);
                            self.backup(&path, fresh.raw_value);
                            return Step::Resolved;
                        }
                    }
                    node = self.edges[e].child as usize;
                    path.push(node as u32);
                }
                State::New => {
                    let candidates = if cfg.tactics {
                        let counts = |h: u64| self.occurrences(h, &path_hashes);
                        match analyse(&pos, &counts) {
                            Analysis::Win(mv) => {
                                self.expand(node, &[mv], &[0.0]);
                                self.nodes[node].raw_value = 1.0;
                                self.nodes[node].proof = -1;
                                self.propagate_proof(&path);
                                self.backup(&path, 1.0);
                                return Step::Resolved;
                            }
                            Analysis::Loss if node != 0 => {
                                let n = &mut self.nodes[node];
                                n.state = State::Terminal;
                                n.raw_value = -1.0;
                                n.proof = 1;
                                self.propagate_proof(&path);
                                self.backup(&path, -1.0);
                                return Step::Resolved;
                            }
                            // A lost root is still searched, to pick the most stubborn move.
                            Analysis::Loss => pos.legal_moves().to_vec(),
                            Analysis::Moves(list) => list.to_vec(),
                        }
                    } else {
                        pos.legal_moves().to_vec()
                    };
                    let mut recent: Vec<Move> = path_moves.iter().rev().copied().collect();
                    recent.extend(self.ctx.recent.iter().copied());
                    recent.truncate(HISTORY_MOVES);
                    let counts = |h: u64| self.occurrences(h, &path_hashes);
                    let state = CompactState::new(&pos, self.ctx.ply + path_moves.len(), &recent, counts);
                    self.nodes[node].state = State::Pending;
                    for &n in &path {
                        self.nodes[n as usize].virtual_loss += 1;
                    }
                    out.push(Leaf { node: node as u32, path, moves: candidates, state });
                    return Step::Queued;
                }
            }
        }
    }

    /// Expands a queued leaf with the network's policy logits (all 4840 actions) and
    /// its value for the side to move.
    pub fn apply(&mut self, leaf: Leaf, policy: &[f32], value: f32, cfg: &SearchConfig, rng: &mut Rng) {
        debug_assert_eq!(policy.len(), NUM_ACTIONS);
        for &n in &leaf.path {
            self.nodes[n as usize].virtual_loss -= 1;
        }
        let inv_t = 1.0 / cfg.policy_temperature.max(1e-3);
        let logits: Vec<f32> = leaf.moves.iter().map(|&m| policy[action_index(m)] * inv_t).collect();
        let node = leaf.node as usize;
        self.expand(node, &leaf.moves, &logits);
        self.nodes[node].raw_value = value;
        if node == 0 {
            self.init_root(cfg, rng);
        }
        self.backup(&leaf.path, value);
    }

    fn expand(&mut self, node: usize, moves: &[Move], logits: &[f32]) {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
        let total: f32 = exps.iter().sum();
        let start = self.edges.len() as u32;
        for ((&mv, &l), &e) in moves.iter().zip(logits).zip(&exps) {
            self.edges.push(Edge { mv, logit: l - max, prior: e / total, child: 0 });
        }
        let n = &mut self.nodes[node];
        n.edge_start = start;
        n.num_edges = moves.len() as u32;
        n.state = State::Expanded;
    }

    fn init_root(&mut self, cfg: &SearchConfig, rng: &mut Rng) {
        let count = self.nodes[0].num_edges as usize;
        self.gumbel = (0..count).map(|_| if cfg.gumbel_noise { rng.gumbel() as f32 } else { 0.0 }).collect();
        let considered = cfg.considered_actions.min(count).max(1);
        // The root evaluation itself is the first visit.
        self.schedule = halving_schedule(considered, self.target.saturating_sub(1).max(1));
        self.forced.clear();
        if cfg.fort_threats && cfg.algorithm == Algorithm::Gumbel && self.root.side == Side::Defenders {
            let root = self.root;
            let counts = |h: u64| self.occurrences(h, &[]);
            self.forced = self.edges_of(0).iter().map(|e| threatens_fort(&root, e.mv, &counts)).collect();
            // At most half of the first visits, to the threats the policy likes best.
            let mut threats: Vec<usize> = (0..count).filter(|&i| self.forced[i]).collect();
            let edges = self.edges_of(0);
            threats.sort_by(|&a, &b| (self.gumbel[b] + edges[b].logit).total_cmp(&(self.gumbel[a] + edges[a].logit)));
            for &i in threats.iter().skip((considered / 2).max(1)) {
                self.forced[i] = false;
            }
        }
    }

    fn backup(&mut self, path: &[u32], leaf_value: f32) {
        let mut v = leaf_value;
        for &n in path.iter().rev() {
            let node = &mut self.nodes[n as usize];
            node.visits += 1;
            node.value_sum -= v;
            v = -v;
        }
    }

    /// Updates proofs from the end of `path` upwards after a node there was proven.
    fn propagate_proof(&mut self, path: &[u32]) {
        for i in (0..path.len().saturating_sub(1)).rev() {
            let parent = path[i] as usize;
            if self.nodes[parent].proof != 0 {
                break;
            }
            let (start, count) = (self.nodes[parent].edge_start as usize, self.nodes[parent].num_edges as usize);
            let mut all_lost = true;
            let mut won = false;
            for edge in &self.edges[start..start + count] {
                let proof = if edge.child == 0 { 0 } else { self.nodes[edge.child as usize].proof };
                won |= proof == 1;
                all_lost &= proof == -1;
            }
            if won {
                self.nodes[parent].proof = -1;
            } else if all_lost {
                self.nodes[parent].proof = 1;
            } else {
                break;
            }
        }
    }

    /// Value of edge `e` for the side to move at its parent, with exact proofs.
    fn edge_q(&self, e: &Edge) -> Option<f32> {
        if e.child == 0 {
            return None;
        }
        let child = &self.nodes[e.child as usize];
        if child.proof != 0 {
            return Some(child.proof as f32);
        }
        (child.visits > 0).then(|| child.value_sum / child.visits as f32)
    }

    fn edge_visits(&self, e: &Edge, with_virtual: bool) -> u32 {
        if e.child == 0 {
            return 0;
        }
        let child = &self.nodes[e.child as usize];
        child.visits + if with_virtual { child.virtual_loss } else { 0 }
    }

    fn pick_edge(&self, node: usize, cfg: &SearchConfig) -> usize {
        let start = self.nodes[node].edge_start as usize;
        let local = match cfg.algorithm {
            Algorithm::Puct => self.pick_puct(node, cfg),
            Algorithm::Gumbel if node == 0 => self.pick_gumbel_root(cfg),
            Algorithm::Gumbel => self.pick_gumbel_interior(node, cfg),
        };
        start + local
    }

    fn edges_of(&self, node: usize) -> &[Edge] {
        let n = &self.nodes[node];
        &self.edges[n.edge_start as usize..(n.edge_start + n.num_edges) as usize]
    }

    fn pick_puct(&self, node: usize, cfg: &SearchConfig) -> usize {
        let n = &self.nodes[node];
        let edges = self.edges_of(node);
        let parent_visits = (n.visits + n.virtual_loss).max(1) as f32;
        let parent_q = if n.visits > 0 { -n.value_sum / n.visits as f32 } else { n.raw_value };
        let visited_policy: f32 = edges.iter().filter(|e| self.edge_visits(e, true) > 0).map(|e| e.prior).sum();
        let fpu = parent_q - cfg.fpu_reduction * visited_policy.sqrt();
        let scale = cfg.c_puct * parent_visits.sqrt();
        let mut best = (f32::NEG_INFINITY, 0);
        for (i, e) in edges.iter().enumerate() {
            let visits = self.edge_visits(e, true);
            let q = if e.child != 0 && self.nodes[e.child as usize].proof != 0 {
                self.nodes[e.child as usize].proof as f32
            } else if visits > 0 {
                let child = &self.nodes[e.child as usize];
                (child.value_sum - child.virtual_loss as f32) / visits as f32
            } else {
                fpu
            };
            let score = q + scale * e.prior / (1.0 + visits as f32);
            if score > best.0 {
                best = (score, i);
            }
        }
        best.1
    }

    /// sigma(completed Q) for every edge of `node` (mctx's qtransform_completed_by_mix_value).
    fn sigma_q(&self, node: usize, cfg: &SearchConfig) -> Vec<f32> {
        let edges = self.edges_of(node);
        let raw = self.nodes[node].raw_value;
        let mut visits_total = 0.0f32;
        let mut prior_visited = 0.0f32;
        let mut weighted = 0.0f32;
        let mut max_visits = 0u32;
        for e in edges {
            let n = self.edge_visits(e, false);
            max_visits = max_visits.max(n);
            if let Some(q) = self.edge_q(e).filter(|_| n > 0) {
                visits_total += n as f32;
                prior_visited += e.prior;
                weighted += e.prior * q;
            }
        }
        let mixed = if prior_visited > 0.0 {
            (raw + visits_total * weighted / prior_visited) / (visits_total + 1.0)
        } else {
            raw
        };
        let mut completed: Vec<f32> =
            edges.iter().map(|e| self.edge_q(e).filter(|_| self.edge_visits(e, false) > 0).unwrap_or(mixed)).collect();
        let lo = completed.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = completed.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = (cfg.maxvisit_init + max_visits as f32) * cfg.value_scale / (hi - lo).max(1e-8);
        for q in &mut completed {
            *q = (*q - lo) * scale;
        }
        completed
    }

    fn pick_gumbel_interior(&self, node: usize, cfg: &SearchConfig) -> usize {
        let edges = self.edges_of(node);
        let sigma = self.sigma_q(node, cfg);
        let scores: Vec<f32> = edges.iter().zip(&sigma).map(|(e, s)| e.logit + s).collect();
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let total: f32 = exps.iter().sum();
        let visits: Vec<u32> = edges.iter().map(|e| self.edge_visits(e, true)).collect();
        let denom = 1.0 + visits.iter().sum::<u32>() as f32;
        let mut best = (f32::NEG_INFINITY, 0);
        for i in 0..edges.len() {
            let score = exps[i] / total - visits[i] as f32 / denom;
            if score > best.0 {
                best = (score, i);
            }
        }
        best.1
    }

    fn pick_gumbel_root(&self, cfg: &SearchConfig) -> usize {
        let edges = self.edges_of(0);
        let visits: Vec<u32> = edges.iter().map(|e| self.edge_visits(e, true)).collect();
        let done: u32 = visits.iter().sum();
        let wanted = match self.schedule.get(done as usize) {
            Some(&v) => v,
            None => *visits.iter().max().unwrap_or(&0),
        };
        let sigma = self.sigma_q(0, cfg);
        // Forced edges take the first of the first visits.
        let forced_first = wanted == 0 && (0..edges.len()).any(|i| visits[i] == 0 && self.is_forced(i));
        let mut best = (f32::NEG_INFINITY, usize::MAX);
        for i in 0..edges.len() {
            if visits[i] != wanted || (forced_first && !self.is_forced(i)) {
                continue;
            }
            let score = self.gumbel[i] + edges[i].logit + sigma[i];
            if score > best.0 {
                best = (score, i);
            }
        }
        if best.1 == usize::MAX {
            // Virtual losses or proven children can leave no action at the scheduled
            // count; fall back to the interior rule.
            return self.pick_gumbel_interior(0, cfg);
        }
        best.1
    }

    fn is_forced(&self, edge: usize) -> bool {
        self.forced.get(edge).copied().unwrap_or(false)
    }

    /// The move to play. Proven wins are played directly (terminal ones first).
    pub fn best_move(&self, cfg: &SearchConfig) -> Move {
        let edges = self.edges_of(0);
        debug_assert!(!edges.is_empty());
        let winning = |e: &&Edge| e.child != 0 && self.nodes[e.child as usize].proof == 1;
        if let Some(e) = edges
            .iter()
            .filter(winning)
            .max_by_key(|e| (self.nodes[e.child as usize].state == State::Terminal, self.edge_visits(e, false)))
        {
            return e.mv;
        }
        let visits: Vec<u32> = edges.iter().map(|e| self.edge_visits(e, false)).collect();
        let most = *visits.iter().max().unwrap();
        let score: Vec<f32> = match cfg.algorithm {
            Algorithm::Gumbel => {
                let sigma = self.sigma_q(0, cfg);
                edges
                    .iter()
                    .enumerate()
                    .map(|(i, e)| self.gumbel.get(i).copied().unwrap_or(0.0) + e.logit + sigma[i])
                    .collect()
            }
            Algorithm::Puct => edges.iter().map(|e| self.edge_q(e).unwrap_or(-2.0)).collect(),
        };
        let mut best = (f32::NEG_INFINITY, 0);
        for i in 0..edges.len() {
            // Never pick a proven loss when anything else is available.
            let lost = edges[i].child != 0 && self.nodes[edges[i].child as usize].proof == -1;
            let s = if visits[i] == most && !lost { score[i] } else { f32::NEG_INFINITY };
            if s > best.0 {
                best = (s, i);
            }
        }
        if best.0 == f32::NEG_INFINITY {
            // Everything most-visited is lost: take the most visited move anyway.
            best.1 = visits.iter().position(|&v| v == most).unwrap();
        }
        edges[best.1].mv
    }

    /// Training target over the root moves: the Gumbel improved policy
    /// softmax(logits + sigma(completed Q)), or visit frequencies for PUCT.
    pub fn policy_target(&self, cfg: &SearchConfig) -> Vec<(Move, f32)> {
        let edges = self.edges_of(0);
        match cfg.algorithm {
            Algorithm::Gumbel => {
                let sigma = self.sigma_q(0, cfg);
                let scores: Vec<f32> = edges.iter().zip(&sigma).map(|(e, s)| e.logit + s).collect();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f32 = exps.iter().sum();
                edges.iter().zip(exps).map(|(e, x)| (e.mv, x / total)).collect()
            }
            Algorithm::Puct => {
                let total: u32 = edges.iter().map(|e| self.edge_visits(e, false)).sum();
                edges.iter().map(|e| (e.mv, self.edge_visits(e, false) as f32 / total.max(1) as f32)).collect()
            }
        }
    }

    /// Search value of the root for its side to move (exact when proven).
    pub fn root_value(&self) -> f32 {
        let root = &self.nodes[0];
        if root.proof != 0 {
            return -(root.proof as f32);
        }
        if root.visits == 0 { 0.0 } else { -root.value_sum / root.visits as f32 }
    }

    /// The root's proven result for its side to move: +1 won, -1 lost, 0 unknown.
    pub fn root_proof(&self) -> i8 {
        -self.nodes[0].proof
    }

    /// (move, visits, Q for the side to move, prior) for every root move.
    pub fn root_stats(&self) -> Vec<(Move, u32, f32, f32)> {
        self.edges_of(0)
            .iter()
            .map(|e| (e.mv, self.edge_visits(e, false), self.edge_q(e).unwrap_or(f32::NAN), e.prior))
            .collect()
    }

    /// Principal variation by most visits.
    pub fn principal_variation(&self, max_len: usize) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut node = 0usize;
        while pv.len() < max_len && self.nodes[node].state == State::Expanded {
            let edges = self.edges_of(node);
            let Some(best) = edges.iter().filter(|e| e.child != 0).max_by_key(|e| self.nodes[e.child as usize].visits)
            else {
                break;
            };
            pv.push(best.mv);
            node = best.child as usize;
        }
        pv
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::NUM_ACTIONS;
    use tafl_core::parse_move;

    /// Runs a search with a uniform network (all logits 0, value 0).
    fn uniform_search(game: &Game, sims: u32, cfg: &SearchConfig) -> Search {
        let mut search = Search::new(game, sims);
        let mut rng = Rng::new(3);
        let policy = vec![0.0f32; NUM_ACTIONS];
        let mut guard = 0;
        while !search.is_complete() {
            let mut leaves = Vec::new();
            match search.select(cfg, &mut leaves) {
                Step::Queued => {
                    for leaf in leaves {
                        search.apply(leaf, &policy, 0.0, cfg, &mut rng);
                    }
                }
                Step::Resolved => {}
                Step::Collision => panic!("no collisions with one leaf at a time"),
            }
            guard += 1;
            assert!(guard < 100_000);
        }
        search
    }

    #[test]
    fn schedule_matches_mctx() {
        // mctx: get_sequence_of_considered_visits(4, 10) == (0, 0, 0, 0, 1, 1, 2, 2, 3, 3)
        assert_eq!(halving_schedule(4, 10), vec![0, 0, 0, 0, 1, 1, 2, 2, 3, 3]);
        assert_eq!(halving_schedule(1, 3), vec![0, 1, 2]);
    }

    #[test]
    fn finds_forced_escape_in_two() {
        // Defenders to move: the king on b6 creates a double threat by moving to the a
        // or k file, opening two corners at once.
        let game = Game::from_fen_and_moves("11/11/11/11/11/1k9/11/11/11/11/9a1 d", &[]).unwrap();
        for algorithm in [Algorithm::Gumbel, Algorithm::Puct] {
            let cfg = SearchConfig { algorithm, gumbel_noise: false, ..SearchConfig::default() };
            let search = uniform_search(&game, 400, &cfg);
            assert_eq!(search.root_proof(), 1, "{algorithm:?}");
            // Several king moves win (to the a or k file, or next to a corner); the
            // chosen one must leave the attackers without a defence.
            let mut after = *game.position();
            after.play(search.best_move(&cfg));
            assert!(matches!(crate::tactics::analyse(&after, &|_| 0), crate::tactics::Analysis::Loss));
        }
    }

    /// From a match against TaflZero (we had the attackers): with this position on the
    /// board, TaflZero's f11f9 left the king two ways into an exit fort.
    const FORT_SETUP: &str = "g11g9 f8d8 k7i7 e7e9 g1g3 f7c7 k8h8 e5e3 f2b2 e9b9 f10b10 f6f10 h11h10 f10c10 b6b8 \
        c10c11 b10b11 c11c9 b11b9 c9c11 b9b11 e6e9 a8a9 c7b7 a4c4 e9b9 g9c9 g7c7 h10c10 b7b9 f11f10 d8d9 c10c9 \
        c11c10 b11b10 c7c8 d11c11 g6g11 i7i9 c10e10 f10j10 d9d11 k6k9 f5f11 i9i11 e3e9 k5i5 g5g10 f1f2 d6d10 a7c7";

    #[test]
    fn fort_threats_get_a_first_visit() {
        let moves: Vec<&str> = FORT_SETUP.split_whitespace().collect();
        let game = Game::from_fen_and_moves(tafl_core::START_FEN, &moves).unwrap();
        let base = SearchConfig { gumbel_noise: false, considered_actions: 8, ..SearchConfig::default() };
        // With a uniform policy only the first 8 moves are considered, and both winning
        // moves (f4f9 and f11f9, the 11th and 71st) are missed.
        let search = uniform_search(&game, 64, &base);
        assert_eq!(search.root_proof(), 0);
        let cfg = SearchConfig { fort_threats: true, ..base };
        let search = uniform_search(&game, 64, &cfg);
        assert_eq!(search.root_proof(), 1);
        let mut after = *game.position();
        after.play(search.best_move(&cfg));
        assert!(matches!(crate::tactics::analyse(&after, &|_| 0), crate::tactics::Analysis::Loss));
    }

    #[test]
    fn repetition_is_seen_on_the_path() {
        // After two shuffles the attackers can win by repeating the position again.
        let mut game = Game::standard();
        for m in ["a4b4", "e5d5", "b4a4", "d5e5", "a4b4", "e5d5", "b4a4", "d5e5"] {
            game.try_play(parse_move(m).unwrap()).unwrap();
        }
        // Consider every root move, so the uniform network does not hide the winner.
        let cfg = SearchConfig { gumbel_noise: false, considered_actions: 256, ..SearchConfig::default() };
        let search = uniform_search(&game, 300, &cfg);
        assert_eq!(search.best_move(&cfg), parse_move("a4b4").unwrap());
        assert_eq!(search.root_proof(), 1);
    }

    #[test]
    fn reroot_keeps_the_subtree() {
        let game = Game::standard();
        let cfg = SearchConfig { algorithm: Algorithm::Puct, ..SearchConfig::default() };
        let search = uniform_search(&game, 2000, &cfg);
        let pv = search.principal_variation(2);
        let mut next = game.clone();
        for &mv in &pv {
            next.try_play(mv).unwrap();
        }
        let mut node = 0usize;
        for &mv in &pv {
            node = search.edges_of(node).iter().find(|e| e.mv == mv).unwrap().child as usize;
        }
        let kept = search.nodes[node].visits;
        let reused = search.reroot(&pv, &next, 100).unwrap();
        assert_eq!(reused.simulations_done(), kept);
        assert_eq!(reused.new_simulations(), 0);
        assert_eq!(reused.target(), kept + 100);
        assert_eq!(reused.root_position().hash, next.position().hash);
        // Every visit below the new root is kept, as one node per visit (plus terminals).
        assert!(reused.node_count() as u32 >= kept && reused.node_count() < search.node_count());
        // The reused search continues to its new target and agrees with the old tree.
        let mut reused = reused;
        let policy = vec![0.0f32; NUM_ACTIONS];
        let mut rng = Rng::new(1);
        while !reused.is_complete() {
            let mut leaves = Vec::new();
            if let Step::Queued = reused.select(&cfg, &mut leaves) {
                for leaf in leaves {
                    reused.apply(leaf, &policy, 0.0, &cfg, &mut rng);
                }
            }
        }
        assert_eq!(reused.simulations_done(), kept + 100);
        assert_eq!(reused.new_simulations(), 100);
        // A reply the old search never tried leaves nothing to reuse.
        let first = search.edges_of(0).iter().find(|e| e.mv == pv[0]).unwrap().child as usize;
        let untried = search.edges_of(first).iter().find(|e| e.child == 0).unwrap().mv;
        let mut other = game.clone();
        other.try_play(pv[0]).unwrap();
        other.try_play(untried).unwrap();
        assert!(search.reroot(&[pv[0], untried], &other, 10).is_none());
    }

    #[test]
    fn policy_target_is_a_distribution() {
        let game = Game::standard();
        for algorithm in [Algorithm::Gumbel, Algorithm::Puct] {
            let cfg = SearchConfig { algorithm, ..SearchConfig::default() };
            let search = uniform_search(&game, 64, &cfg);
            let target = search.policy_target(&cfg);
            assert_eq!(target.len(), 116);
            let total: f32 = target.iter().map(|t| t.1).sum();
            assert!((total - 1.0).abs() < 1e-4);
            assert_eq!(search.simulations_done(), 64);
        }
    }
}
