//! C ABI for the Python side (loaded with ctypes).
//!
//! Conventions: functions returning `i32` give a non-negative count on success and
//! -1 on failure, with the message available from `tafl_last_error`. Text results are
//! copied into caller buffers; the returned length is the full length (call again
//! with a larger buffer when it exceeds the capacity).
//!
//! # Safety
//!
//! Every function trusts its caller (the Python bindings): handles come from
//! `tafl_pool_new` and are not used after `tafl_pool_free`, a handle is used by one
//! thread at a time, strings are NUL-terminated, and buffers hold at least the
//! documented number of elements.
#![allow(clippy::missing_safety_doc)]

use std::cell::RefCell;
use std::ffi::{CStr, c_char};

use rayon::prelude::*;
use serde_json::{Value, json};
use tafl_core::symmetry::NUM_SYMMETRIES;
use tafl_core::{Game, Move, Side, parse_move};
use tafl_engine::encode::{
    COMPACT_STATE_BYTES, CompactState, FEATURE_BYTES, NUM_ACTIONS, NUM_PLANES, action_index, symmetry_action_table,
};
use tafl_engine::mcts::{Algorithm, SearchConfig};
use tafl_engine::pool::{FinishedGame, Pool, PoolConfig, SAMPLE_RECORD_BYTES, SampleRecord};

pub const ABI_VERSION: u32 = 1;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn fail(message: impl Into<String>) -> i32 {
    LAST_ERROR.with(|e| *e.borrow_mut() = message.into());
    -1
}

/// Copies `text` (NUL-terminated when it fits) and returns its length.
unsafe fn write_text(text: &str, buf: *mut c_char, cap: usize) -> usize {
    if !buf.is_null() && cap > 0 {
        let n = text.len().min(cap - 1);
        unsafe {
            std::ptr::copy_nonoverlapping(text.as_ptr(), buf as *mut u8, n);
            *buf.add(n) = 0;
        }
    }
    text.len()
}

unsafe fn read_str<'a>(ptr: *const c_char) -> Result<&'a str, String> {
    if ptr.is_null() {
        return Err("null string".into());
    }
    unsafe { CStr::from_ptr(ptr) }.to_str().map_err(|e| format!("invalid UTF-8: {e}"))
}

#[unsafe(no_mangle)]
pub extern "C" fn tafl_abi_version() -> u32 {
    ABI_VERSION
}

/// Layout constants: planes, actions, feature bytes, compact state bytes, sample
/// record bytes, symmetries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tafl_constants(out: *mut u32, n: usize) -> i32 {
    let values = [
        NUM_PLANES as u32,
        NUM_ACTIONS as u32,
        FEATURE_BYTES as u32,
        COMPACT_STATE_BYTES as u32,
        SAMPLE_RECORD_BYTES as u32,
        NUM_SYMMETRIES as u32,
    ];
    let n = n.min(values.len());
    unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), out, n) };
    values.len() as i32
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tafl_last_error(buf: *mut c_char, cap: usize) -> usize {
    LAST_ERROR.with(|e| unsafe { write_text(&e.borrow(), buf, cap) })
}

/// `NUM_SYMMETRIES * NUM_ACTIONS` entries: the action each action becomes under each
/// symmetry.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tafl_symmetry_actions(out: *mut u16) {
    for (t, table) in symmetry_action_table().iter().enumerate() {
        unsafe { std::ptr::copy_nonoverlapping(table.as_ptr(), out.add(t * NUM_ACTIONS), NUM_ACTIONS) };
    }
}

/// Input planes for `n` compact states, each under its own symmetry.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tafl_encode_states(states: *const CompactState, syms: *const u8, n: usize, out: *mut u8) {
    let states = unsafe { std::slice::from_raw_parts(states, n) };
    let syms = unsafe { std::slice::from_raw_parts(syms, n) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, n * FEATURE_BYTES) };
    out.par_chunks_mut(FEATURE_BYTES).zip(states.par_iter().zip(syms)).for_each(|(chunk, (state, &t))| {
        state.write_planes(t as usize % NUM_SYMMETRIES, chunk);
    });
}

fn parse_search(v: &Value, cfg: &mut SearchConfig) -> Result<(), String> {
    let Some(map) = v.as_object() else { return Err("search config must be an object".into()) };
    for (key, value) in map {
        let num = || value.as_f64().ok_or_else(|| format!("search.{key} must be a number"));
        let flag = || value.as_bool().ok_or_else(|| format!("search.{key} must be a boolean"));
        match key.as_str() {
            "algorithm" => {
                cfg.algorithm = match value.as_str() {
                    Some("gumbel") => Algorithm::Gumbel,
                    Some("puct") => Algorithm::Puct,
                    _ => return Err("search.algorithm must be \"gumbel\" or \"puct\"".into()),
                }
            }
            "considered_actions" => cfg.considered_actions = num()? as usize,
            "gumbel_noise" => cfg.gumbel_noise = flag()?,
            "value_scale" => cfg.value_scale = num()? as f32,
            "maxvisit_init" => cfg.maxvisit_init = num()? as f32,
            "c_puct" => cfg.c_puct = num()? as f32,
            "fpu_reduction" => cfg.fpu_reduction = num()? as f32,
            "policy_temperature" => cfg.policy_temperature = num()? as f32,
            "tactics" => cfg.tactics = flag()?,
            "fort_threats" => cfg.fort_threats = flag()?,
            other => return Err(format!("unknown search option '{other}'")),
        }
    }
    Ok(())
}

fn parse_pool_config(text: &str) -> Result<PoolConfig, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("config is not JSON: {e}"))?;
    let Some(map) = v.as_object() else { return Err("config must be a JSON object".into()) };
    let mut cfg = PoolConfig::default();
    for (key, value) in map {
        let num = || value.as_f64().ok_or_else(|| format!("{key} must be a number"));
        match key.as_str() {
            "games" => cfg.games = num()? as usize,
            "search" => parse_search(value, &mut cfg.search)?,
            "full_simulations" => cfg.full_simulations = num()? as u32,
            "fast_simulations" => cfg.fast_simulations = num()? as u32,
            "full_search_prob" => cfg.full_search_prob = num()?,
            "leaves_per_search" => cfg.leaves_per_search = num()? as usize,
            "max_plies" => cfg.max_plies = num()? as usize,
            "seed" => cfg.seed = num()? as u64,
            "resign_threshold" => cfg.resign_threshold = num()? as f32,
            "resign_moves" => cfg.resign_moves = num()? as u32,
            "no_resign_fraction" => cfg.no_resign_fraction = num()?,
            "autoplay" => cfg.autoplay = value.as_bool().ok_or("autoplay must be a boolean")?,
            "reuse_tree" => cfg.reuse_tree = value.as_bool().ok_or("reuse_tree must be a boolean")?,
            other => return Err(format!("unknown pool option '{other}'")),
        }
    }
    if cfg.games == 0 || cfg.leaves_per_search == 0 {
        return Err("games and leaves_per_search must be positive".into());
    }
    Ok(cfg)
}

/// Finished games waiting to be copied out.
pub struct Handle {
    pool: Pool,
    outbox: Vec<FinishedGame>,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tafl_pool_new(config_json: *const c_char) -> *mut Handle {
    let result = unsafe { read_str(config_json) }.and_then(parse_pool_config);
    match result {
        Ok(cfg) => Box::into_raw(Box::new(Handle { pool: Pool::new(cfg), outbox: Vec::new() })),
        Err(e) => {
            fail(e);
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tafl_pool_free(handle: *mut Handle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tafl_pool_max_batch(handle: *const Handle) -> u32 {
    unsafe { &*handle }.pool.max_batch() as u32
}

/// Selection phase; writes `FEATURE_BYTES` per row into `planes` (capacity in rows).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tafl_pool_collect(handle: *mut Handle, planes: *mut u8, capacity: usize) -> i32 {
    let h = unsafe { &mut *handle };
    if capacity < h.pool.max_batch() {
        return fail(format!("plane buffer holds {capacity} rows, need {}", h.pool.max_batch()));
    }
    let planes = unsafe { std::slice::from_raw_parts_mut(planes, capacity * FEATURE_BYTES) };
    let rows = h.pool.collect(planes);
    h.outbox.extend(h.pool.take_finished());
    rows as i32
}

/// Evaluation phase: `rows` rows of `NUM_ACTIONS` policy logits and one value each.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tafl_pool_apply(
    handle: *mut Handle,
    policy: *const f32,
    value: *const f32,
    rows: usize,
) -> i32 {
    let h = unsafe { &mut *handle };
    let policy = unsafe { std::slice::from_raw_parts(policy, rows * NUM_ACTIONS) };
    let value = unsafe { std::slice::from_raw_parts(value, rows) };
    h.pool.apply(policy, value);
    0
}

/// Counts of finished games, samples and policy entries waiting in the outbox.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tafl_pool_pending(
    handle: *const Handle,
    games: *mut u64,
    samples: *mut u64,
    entries: *mut u64,
) {
    let h = unsafe { &*handle };
    let s: usize = h.outbox.iter().map(|g| g.samples.len()).sum();
    let e: usize = h.outbox.iter().flat_map(|g| g.samples.iter()).map(|(_, p)| p.len()).sum();
    unsafe {
        *games = h.outbox.len() as u64;
        *samples = s as u64;
        *entries = e as u64;
    }
}

/// Moves the outbox into caller buffers sized from `tafl_pool_pending`: sample records,
/// policy entries (actions and probabilities, concatenated in record order) and one
/// JSON line per game. Returns the JSON length (-1 if the text buffer is too small,
/// in which case nothing is taken).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tafl_pool_take(
    handle: *mut Handle,
    records: *mut SampleRecord,
    actions: *mut u16,
    probs: *mut f32,
    text: *mut c_char,
    text_cap: usize,
) -> i32 {
    let h = unsafe { &mut *handle };
    let mut lines = String::new();
    for game in &h.outbox {
        let s = &game.summary;
        let moves: Vec<String> = s.moves.iter().map(Move::to_string).collect();
        let (winner, reason) = match s.outcome {
            Some(o) => (Some(side_name(o.winner)), Some(format!("{:?}", o.reason))),
            None => (s.resigned_by.map(|r| side_name(r.other())), s.resigned_by.map(|_| "Resignation".to_string())),
        };
        let line = json!({
            "id": s.id, "start": s.start_fen, "moves": moves.join(" "), "plies": s.moves.len(),
            "winner": winner, "reason": reason, "no_resign": s.no_resign,
            "would_resign": s.would_resign.map(side_name), "samples": game.samples.len(),
        });
        lines.push_str(&line.to_string());
        lines.push('\n');
    }
    if lines.len() + 1 > text_cap {
        return fail(format!("text buffer of {text_cap} bytes is too small for {}", lines.len() + 1));
    }
    unsafe { write_text(&lines, text, text_cap) };
    let (mut r, mut e) = (0usize, 0usize);
    for game in h.outbox.drain(..) {
        for (record, policy) in game.samples {
            unsafe { *records.add(r) = record };
            r += 1;
            for (action, prob) in policy {
                unsafe {
                    *actions.add(e) = action;
                    *probs.add(e) = prob;
                }
                e += 1;
            }
        }
    }
    lines.len() as i32
}

fn side_name(side: Side) -> &'static str {
    match side {
        Side::Attackers => "attackers",
        Side::Defenders => "defenders",
    }
}

fn parse_game(fen: &str, moves: &str) -> Result<Game, String> {
    let fen = if fen.trim().is_empty() || fen.trim() == "startpos" { tafl_core::START_FEN } else { fen };
    let moves: Vec<&str> = moves.split_whitespace().collect();
    Game::from_fen_and_moves(fen, &moves)
}

/// Analysis mode: sets slot `slot` to the game `fen` (or "startpos") plus `moves`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tafl_pool_set_game(
    handle: *mut Handle,
    slot: usize,
    fen: *const c_char,
    moves: *const c_char,
) -> i32 {
    let h = unsafe { &mut *handle };
    if slot >= h.pool.len() {
        return fail("slot out of range");
    }
    let parsed = unsafe { read_str(fen) }.and_then(|f| unsafe { read_str(moves) }.and_then(|m| parse_game(f, m)));
    match parsed {
        Ok(game) => {
            h.pool.set_game(slot, game);
            0
        }
        Err(e) => fail(e),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tafl_pool_play(handle: *mut Handle, slot: usize, mv: *const c_char) -> i32 {
    let h = unsafe { &mut *handle };
    if slot >= h.pool.len() {
        return fail("slot out of range");
    }
    let parsed = unsafe { read_str(mv) }.and_then(|m| parse_move(m).ok_or_else(|| format!("malformed move '{m}'")));
    match parsed.and_then(|m| h.pool.play(slot, m)) {
        Ok(()) => 0,
        Err(e) => fail(e),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tafl_pool_start_search(handle: *mut Handle, slot: usize, simulations: u32) -> i32 {
    let h = unsafe { &mut *handle };
    if slot >= h.pool.len() {
        return fail("slot out of range");
    }
    if h.pool.game(slot).is_over() {
        return fail("the game is over");
    }
    h.pool.start_search(slot, simulations);
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tafl_pool_stop_search(handle: *mut Handle, slot: usize) -> i32 {
    let h = unsafe { &mut *handle };
    if slot >= h.pool.len() {
        return fail("slot out of range");
    }
    h.pool.stop_search(slot);
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tafl_pool_is_thinking(handle: *const Handle, slot: usize) -> i32 {
    let h = unsafe { &*handle };
    i32::from(slot < h.pool.len() && h.pool.is_thinking(slot))
}

/// JSON with the game state of a slot and its latest search result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tafl_pool_state(handle: *const Handle, slot: usize, buf: *mut c_char, cap: usize) -> usize {
    let h = unsafe { &*handle };
    let text = if slot >= h.pool.len() {
        json!({"error": "slot out of range"}).to_string()
    } else {
        let mut v = game_json(h.pool.game(slot));
        if let Some(r) = h.pool.result(slot) {
            let stats: Vec<Value> = r
                .stats
                .iter()
                .map(|(m, n, q, p)| json!({"move": m.to_string(), "visits": n, "q": if q.is_nan() { Value::Null } else { json!(q) }, "prior": p}))
                .collect();
            let policy: Vec<Value> = r.policy.iter().map(|(m, p)| json!([m.to_string(), p])).collect();
            v["result"] = json!({
                "best": r.best.to_string(), "value": r.value, "proof": r.proof, "simulations": r.simulations,
                "new_simulations": r.new_simulations,
                "pv": r.pv.iter().map(Move::to_string).collect::<Vec<_>>(), "stats": stats, "policy": policy,
            });
        }
        v.to_string()
    };
    unsafe { write_text(&text, buf, cap) }
}

fn game_json(game: &Game) -> Value {
    let pos = game.position();
    let outcome = game.outcome().map(|o| json!({"winner": side_name(o.winner), "reason": format!("{:?}", o.reason)}));
    json!({
        "fen": pos.fen(),
        "side": side_name(pos.side),
        "ply": game.moves().len(),
        "moves": game.moves().iter().map(Move::to_string).collect::<Vec<_>>(),
        "legal": if game.is_over() { vec![] } else { pos.legal_moves().iter().map(Move::to_string).collect::<Vec<_>>() },
        "outcome": outcome,
    })
}

/// Referee: JSON for the game `fen` (or "startpos") plus `moves`, or an error object.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tafl_game_json(
    fen: *const c_char,
    moves: *const c_char,
    buf: *mut c_char,
    cap: usize,
) -> usize {
    let parsed = unsafe { read_str(fen) }.and_then(|f| unsafe { read_str(moves) }.and_then(|m| parse_game(f, m)));
    let text = match parsed {
        Ok(game) => game_json(&game).to_string(),
        Err(e) => json!({"error": e}).to_string(),
    };
    unsafe { write_text(&text, buf, cap) }
}

/// Action index of a move string (-1 if malformed).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tafl_action_index(mv: *const c_char) -> i32 {
    match unsafe { read_str(mv) }.ok().and_then(parse_move) {
        Some(m) => action_index(m) as i32,
        None => fail("malformed move"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_parsing() {
        let cfg =
            parse_pool_config(r#"{"games": 8, "search": {"algorithm": "puct", "c_puct": 2.0}, "autoplay": false}"#)
                .unwrap();
        assert_eq!(cfg.games, 8);
        assert_eq!(cfg.search.algorithm, Algorithm::Puct);
        assert!(!cfg.autoplay);
        assert!(parse_pool_config(r#"{"gamez": 1}"#).is_err());
        assert!(parse_pool_config(r#"{"search": {"algorithm": "mcts"}}"#).is_err());
    }
}
