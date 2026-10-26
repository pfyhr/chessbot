//! The browser build: a flat C ABI over `az-core`, driven from JavaScript.
//!
//! No `wasm-bindgen`. The surface is narrow enough -- a position, a resumable
//! search, and a scratch buffer for strings -- that hand-written glue is less
//! machinery than a code generator, and it keeps the CI build to `cargo build
//! --target wasm32-unknown-unknown` with no extra toolchain to pin.
//!
//! # Why the search crosses the boundary at all
//!
//! The network runs in JavaScript (ONNX Runtime Web), and its inference is
//! asynchronous. [`az_core::mcts::Search`] is already written as a resumable
//! state machine for exactly this reason -- self-play pools leaf evaluations
//! from many games into one batch -- so the browser drives the same loop the
//! trainer does:
//!
//! ```text
//! eng_search_start(h, sims, considered, seed);
//! while eng_search_prepare(h) == 0 {
//!     let obs = eng_search_encode(h);      // 119*64 floats in wasm memory
//!     let (logits, value) = await onnx(obs);
//!     eng_search_apply(h, logits, value);
//! }
//! ```
//!
//! That is the point of the whole exercise: the page plays with the *same*
//! Gumbel MCTS, the same encoding and the same move indexing as training, not a
//! reimplementation that agrees on the easy positions.
//!
//! # Two rules for the JavaScript side
//!
//! 1. **Re-read `memory.buffer` after every call.** Allocating inside wasm can
//!    grow the memory, which detaches every existing `TypedArray` view.
//! 2. **String results are returned as a length, into a shared scratch buffer.**
//!    Call the function (which returns the byte length), then
//!    [`eng_str_ptr`] -- in that order, because filling the buffer may move it.

use az_core::chess::ChessPos;
use az_core::game::{obs_len, Game, Outcome};
use az_core::mcts::{Config, Rng, Search, Status};
use shakmaty::san::SanPlus;
use shakmaty::uci::UciMove;
use shakmaty::{Move, Position};

const OBS: usize = obs_len::<ChessPos>();
const POLICY: usize = ChessPos::POLICY_LEN;

/// Game state plus whatever search is in flight.
pub struct Engine {
    /// Every position from the start, so take-back is a pop rather than an
    /// unmake -- `ChessPos` carries an `Arc` history chain and has no inverse.
    stack: Vec<ChessPos>,
    played: Vec<String>,
    /// SAN for each played move, recorded at play time.
    ///
    /// SAN depends on the position *before* the move -- which pieces could also
    /// have reached the square, whether it gives check -- so it cannot be
    /// recovered from a UCI list alone without replaying the game.
    san: Vec<String>,
    search: Option<Search<ChessPos>>,
    obs: Vec<f32>,
    strbuf: Vec<u8>,
}

impl Engine {
    fn cur(&self) -> &ChessPos {
        self.stack.last().expect("stack is never empty")
    }

    fn resolve(&self, uci: &str) -> Option<Move> {
        uci.parse::<UciMove>()
            .ok()?
            .to_move(self.cur().inner())
            .ok()
    }

    fn set_str(&mut self, s: &str) -> usize {
        self.strbuf.clear();
        self.strbuf.extend_from_slice(s.as_bytes());
        self.strbuf.len()
    }
}

// ---------------------------------------------------------------------------
// Raw memory, for handing FENs and logits across the boundary
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wasm_alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// # Safety
/// `ptr` must come from [`wasm_alloc`] with the same `len`.
#[no_mangle]
pub unsafe extern "C" fn wasm_free(ptr: *mut u8, len: usize) {
    drop(Vec::from_raw_parts(ptr, 0, len));
}

#[no_mangle]
pub extern "C" fn obs_floats() -> usize {
    OBS
}

#[no_mangle]
pub extern "C" fn policy_floats() -> usize {
    POLICY
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn eng_new() -> *mut Engine {
    Box::into_raw(Box::new(Engine {
        stack: vec![ChessPos::new()],
        played: Vec::new(),
        san: Vec::new(),
        search: None,
        obs: vec![0.0; OBS],
        strbuf: Vec::new(),
    }))
}

/// # Safety
/// `h` must come from [`eng_new`] and must not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn eng_free(h: *mut Engine) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

macro_rules! eng {
    ($h:expr) => {
        match $h.as_mut() {
            Some(e) => e,
            None => return Default::default(),
        }
    };
}

// ---------------------------------------------------------------------------
// Position
// ---------------------------------------------------------------------------

/// Reset to the standard starting position.
///
/// # Safety
/// `h` must be a live engine handle.
#[no_mangle]
pub unsafe extern "C" fn eng_reset(h: *mut Engine) {
    let e = eng!(h);
    e.stack.truncate(1);
    e.stack[0] = ChessPos::new();
    e.played.clear();
    e.san.clear();
    e.search = None;
}

/// Replace the game with a position parsed from FEN, discarding history.
///
/// Returns 0 on success, -1 if the FEN is unparseable or illegal. Note that a
/// position set this way has no move history, so threefold repetition cannot be
/// detected until three positions have been played from it.
///
/// # Safety
/// `h` must be a live engine handle; `ptr`/`len` a valid UTF-8 byte range.
#[no_mangle]
pub unsafe extern "C" fn eng_set_fen(h: *mut Engine, ptr: *const u8, len: usize) -> i32 {
    let e = eng!(h);
    let Ok(fen) = std::str::from_utf8(std::slice::from_raw_parts(ptr, len)) else {
        return -1;
    };
    match ChessPos::from_fen(fen) {
        Ok(p) => {
            e.stack.clear();
            e.stack.push(p);
            e.played.clear();
            e.san.clear();
            e.search = None;
            0
        }
        Err(_) => -1,
    }
}

/// Fill the scratch buffer with the current FEN; returns its byte length.
///
/// # Safety
/// `h` must be a live engine handle.
#[no_mangle]
pub unsafe extern "C" fn eng_fen(h: *mut Engine) -> usize {
    let e = eng!(h);
    let s = e.cur().fen();
    e.set_str(&s)
}

/// Pointer to the scratch buffer. Call *after* whichever function filled it.
///
/// # Safety
/// `h` must be a live engine handle.
#[no_mangle]
pub unsafe extern "C" fn eng_str_ptr(h: *mut Engine) -> *const u8 {
    match h.as_ref() {
        Some(e) => e.strbuf.as_ptr(),
        None => std::ptr::null(),
    }
}

/// 0 if White is to move, 1 if Black.
///
/// # Safety
/// `h` must be a live engine handle.
#[no_mangle]
pub unsafe extern "C" fn eng_turn(h: *mut Engine) -> i32 {
    let e = eng!(h);
    match e.cur().player_to_move() {
        az_core::Player::First => 0,
        az_core::Player::Second => 1,
    }
}

/// Plies played from the position the game started at.
///
/// # Safety
/// `h` must be a live engine handle.
#[no_mangle]
pub unsafe extern "C" fn eng_ply(h: *mut Engine) -> usize {
    let e = eng!(h);
    e.played.len()
}

/// Space-separated legal moves in UCI, into the scratch buffer.
///
/// # Safety
/// `h` must be a live engine handle.
#[no_mangle]
pub unsafe extern "C" fn eng_legal_uci(h: *mut Engine) -> usize {
    let e = eng!(h);
    let mut buf = Vec::new();
    e.cur().legal_moves(&mut buf);
    let s = buf
        .into_iter()
        .map(|m| UciMove::from_standard(m).to_string())
        .collect::<Vec<_>>()
        .join(" ");
    e.set_str(&s)
}

/// Terminal status: -1 still going, 0 draw, 1 side-to-move wins, 2 it loses.
///
/// Mirrors [`az_core::game::Game::expand`], so a fifty-move or repetition draw
/// reports as terminal even though legal moves remain.
///
/// # Safety
/// `h` must be a live engine handle.
#[no_mangle]
pub unsafe extern "C" fn eng_outcome(h: *mut Engine) -> i32 {
    let e = eng!(h);
    match e.cur().outcome() {
        None => -1,
        Some(Outcome::Draw) => 0,
        Some(Outcome::Win) => 1,
        Some(Outcome::Loss) => 2,
    }
}

/// Why the game ended: 0 not over, 1 checkmate, 2 stalemate, 3 insufficient
/// material, 4 fifty-move rule, 5 threefold repetition.
///
/// The order follows `az-core`'s own precedence in `Game::expand`, so this can
/// never disagree with [`eng_outcome`] about whether the game is over -- only
/// elaborate on it. A page that can say *stalemate* rather than *draw* is the
/// difference between a result and a shrug.
///
/// # Safety
/// `h` must be a live engine handle.
#[no_mangle]
pub unsafe extern "C" fn eng_outcome_reason(h: *mut Engine) -> i32 {
    let e = eng!(h);
    let pos = e.cur();
    let mut moves = Vec::new();
    if pos.expand(&mut moves).is_none() {
        return 0;
    }
    if moves.is_empty() {
        return if pos.inner().is_check() { 1 } else { 2 };
    }
    if pos.inner().is_insufficient_material() {
        return 3;
    }
    if pos.inner().halfmoves() >= 100 {
        return 4;
    }
    5
}

/// # Safety
/// `h` must be a live engine handle.
#[no_mangle]
pub unsafe extern "C" fn eng_in_check(h: *mut Engine) -> i32 {
    let e = eng!(h);
    e.cur().inner().is_check() as i32
}

/// Play a UCI move. Returns 0, or -1 if it is not legal here.
///
/// # Safety
/// `h` must be a live engine handle; `ptr`/`len` a valid UTF-8 byte range.
#[no_mangle]
pub unsafe extern "C" fn eng_play_uci(h: *mut Engine, ptr: *const u8, len: usize) -> i32 {
    let e = eng!(h);
    let Ok(uci) = std::str::from_utf8(std::slice::from_raw_parts(ptr, len)) else {
        return -1;
    };
    let Some(mv) = e.resolve(uci) else {
        return -1;
    };
    // Before the move is made: SAN is defined against the prior position.
    let san = SanPlus::from_move(e.cur().inner().clone(), mv).to_string();
    let mut next = e.cur().clone();
    next.play(mv);
    e.stack.push(next);
    e.san.push(san);
    // Re-spelled from the resolved move so the stored list is canonical: a GUI
    // may send "e1h1" for a castle, and the move list should not show that.
    e.played.push(UciMove::from_standard(mv).to_string());
    e.search = None;
    0
}

/// Take back one ply. Returns 0, or -1 at the start of the game.
///
/// # Safety
/// `h` must be a live engine handle.
#[no_mangle]
pub unsafe extern "C" fn eng_undo(h: *mut Engine) -> i32 {
    let e = eng!(h);
    if e.stack.len() < 2 {
        return -1;
    }
    e.stack.pop();
    e.played.pop();
    e.san.pop();
    e.search = None;
    0
}

/// Space-separated UCI of every move played, into the scratch buffer.
///
/// # Safety
/// `h` must be a live engine handle.
#[no_mangle]
pub unsafe extern "C" fn eng_history_uci(h: *mut Engine) -> usize {
    let e = eng!(h);
    let s = e.played.join(" ");
    e.set_str(&s)
}

/// Space-separated SAN of every move played, into the scratch buffer.
///
/// # Safety
/// `h` must be a live engine handle.
#[no_mangle]
pub unsafe extern "C" fn eng_history_san(h: *mut Engine) -> usize {
    let e = eng!(h);
    let s = e.san.join(" ");
    e.set_str(&s)
}

/// Flat policy index of a UCI move in this position, or -1 if it is illegal.
///
/// Exposed for the parity harness: the index is the one place where the browser
/// and the trainer could silently disagree and still both look like chess.
///
/// # Safety
/// `h` must be a live engine handle; `ptr`/`len` a valid UTF-8 byte range.
#[no_mangle]
pub unsafe extern "C" fn eng_policy_index(h: *mut Engine, ptr: *const u8, len: usize) -> i32 {
    let e = eng!(h);
    let Ok(uci) = std::str::from_utf8(std::slice::from_raw_parts(ptr, len)) else {
        return -1;
    };
    match e.resolve(uci) {
        Some(mv) => e.cur().policy_index(mv) as i32,
        None => -1,
    }
}

/// Encode the current position into the observation buffer; returns its pointer.
///
/// # Safety
/// `h` must be a live engine handle.
#[no_mangle]
pub unsafe extern "C" fn eng_encode(h: *mut Engine) -> *const f32 {
    let Some(e) = h.as_mut() else {
        return std::ptr::null();
    };
    e.obs.resize(OBS, 0.0);
    let mut obs = std::mem::take(&mut e.obs);
    e.cur().encode(&mut obs);
    e.obs = obs;
    e.obs.as_ptr()
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// Begin a search from the current position.
///
/// # Safety
/// `h` must be a live engine handle.
#[no_mangle]
pub unsafe extern "C" fn eng_search_start(
    h: *mut Engine,
    sims: u32,
    considered: usize,
    seed: u64,
) {
    let e = eng!(h);
    let cfg = Config {
        sims,
        max_considered: considered,
        ..Config::default()
    };
    // One `Rng` per search, seeded explicitly -- same construction as the
    // Python bridge, so a fixed seed gives the same Gumbel draws in both.
    let mut rng = Rng::new(seed);
    e.search = Some(Search::new(e.cur().clone(), cfg, &mut rng));
}

/// Advance the search. Returns 0 if it needs an evaluation, 1 if it is done.
///
/// # Safety
/// `h` must be a live engine handle with a started search.
#[no_mangle]
pub unsafe extern "C" fn eng_search_prepare(h: *mut Engine) -> i32 {
    let e = eng!(h);
    match e.search.as_mut() {
        Some(s) => match s.prepare() {
            Status::NeedsEval => 0,
            Status::Complete => 1,
        },
        None => 1,
    }
}

/// Encode the position awaiting evaluation; returns the observation pointer.
///
/// # Safety
/// `h` must be a live engine handle with a started search.
#[no_mangle]
pub unsafe extern "C" fn eng_search_encode(h: *mut Engine) -> *const f32 {
    let Some(e) = h.as_mut() else {
        return std::ptr::null();
    };
    let Some(s) = e.search.as_ref() else {
        return std::ptr::null();
    };
    e.obs.resize(OBS, 0.0);
    let mut obs = std::mem::take(&mut e.obs);
    s.pending().encode(&mut obs);
    e.obs = obs;
    e.obs.as_ptr()
}

/// Hand back the network's answer: `POLICY` logits and a scalar value.
///
/// `value` is `P(win) - P(loss)` from the evaluated position's mover's point of
/// view, matching what the trainer backs up.
///
/// # Safety
/// `h` must be a live engine handle; `logits` must point to [`policy_floats`]
/// readable `f32`s.
#[no_mangle]
pub unsafe extern "C" fn eng_search_apply(h: *mut Engine, logits: *const f32, value: f32) {
    let e = eng!(h);
    let Some(s) = e.search.as_mut() else { return };
    s.apply(std::slice::from_raw_parts(logits, POLICY), value);
}

/// The move the search settled on, into the scratch buffer.
///
/// # Safety
/// `h` must be a live engine handle with a completed search.
#[no_mangle]
pub unsafe extern "C" fn eng_search_best(h: *mut Engine) -> usize {
    let e = eng!(h);
    let Some(s) = e.search.as_ref() else { return 0 };
    let s = UciMove::from_standard(s.result().0).to_string();
    e.set_str(&s)
}

/// Root value after search, from the root mover's point of view.
///
/// # Safety
/// `h` must be a live engine handle with a completed search.
#[no_mangle]
pub unsafe extern "C" fn eng_search_value(h: *mut Engine) -> f32 {
    let e = eng!(h);
    e.search.as_ref().map(|s| s.root_value()).unwrap_or(0.0)
}

/// Simulations actually spent. Lower than requested when the root is shallow.
///
/// # Safety
/// `h` must be a live engine handle with a started search.
#[no_mangle]
pub unsafe extern "C" fn eng_search_sims(h: *mut Engine) -> u32 {
    let e = eng!(h);
    e.search.as_ref().map(|s| s.simulations()).unwrap_or(0)
}

/// A result the rules guarantee at the root: -1 none, 0 draw, 1 win, 2 loss.
///
/// # Safety
/// `h` must be a live engine handle with a started search.
#[no_mangle]
pub unsafe extern "C" fn eng_search_proven(h: *mut Engine) -> i32 {
    let e = eng!(h);
    match e.search.as_ref().and_then(|s| s.proven()) {
        None => -1,
        Some(Outcome::Draw) => 0,
        Some(Outcome::Win) => 1,
        Some(Outcome::Loss) => 2,
    }
}

/// Per-root-move detail as JSON, best first, into the scratch buffer.
///
/// Each entry is `{m, p, prior, n, q}` -- move, improved policy, raw network
/// prior, visits, mean value. The gap between `p` and `prior` is exactly how
/// much the search disagreed with the network, which is the interesting part to
/// show a person.
///
/// # Safety
/// `h` must be a live engine handle with a completed search.
#[no_mangle]
pub unsafe extern "C" fn eng_search_detail(h: *mut Engine, top: usize) -> usize {
    let e = eng!(h);
    let Some(s) = e.search.as_ref() else {
        return e.set_str("[]");
    };
    let rows = s.root_detail();
    let mut out = String::from("[");
    for (i, (mv, p, prior, visits, q, proven)) in rows.iter().take(top).enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"m\":\"{}\",\"p\":{},\"prior\":{},\"n\":{},\"q\":{},\"proven\":{}}}",
            UciMove::from_standard(*mv),
            json_f32(Some(*p)),
            json_f32(Some(*prior)),
            visits,
            json_f32(*q),
            match proven {
                None => "null".to_string(),
                Some(Outcome::Draw) => "0".to_string(),
                Some(Outcome::Win) => "1".to_string(),
                Some(Outcome::Loss) => "2".to_string(),
            }
        ));
    }
    out.push(']');
    e.set_str(&out)
}

/// `null` rather than a bare `NaN`, which is not JSON and would throw in the
/// page's `JSON.parse` instead of merely looking wrong.
fn json_f32(v: Option<f32>) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{x:.6}"),
        _ => "null".to_string(),
    }
}
