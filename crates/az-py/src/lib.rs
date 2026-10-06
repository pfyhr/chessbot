//! Python bindings for the AlphaZero core.
//!
//! # Where the boundary sits
//!
//! Python drives the training loop, but Rust owns every board and every search
//! tree. The boundary is crossed **once per neural-net batch**, never once per
//! node. That is the whole reason the hot path is in Rust at all: a per-node
//! crossing would cost more than the movegen it wraps.
//!
//! Measured at 0.0-2.5% of encode-plus-forward, so the design has roughly 40x of
//! headroom -- see `bench/results/2026-08-16-bridge-throughput.md`.
//!
//! # Move strings
//!
//! Chess moves cross as UCI strings (`e2e4`, `e7e8q`, `e1g1`). Castling is the
//! king's two-square move, matching python-chess, not shakmaty's internal
//! king-takes-rook form. Connect4 moves are plain 0-indexed column numbers.

mod chess;
mod connect4;

use az_core::game::Game;
use az_core::ChessPos;
use pyo3::prelude::*;

/// TEMPORARY (2026-10-06): hot-path counters. Empty unless az-core/hotprof is on.
#[pyfunction]
fn hot_profile() -> Vec<(String, u64)> {
    az_core::prof::snapshot()
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
}

/// TEMPORARY (2026-10-06): zero the hot-path counters.
#[pyfunction]
fn hot_profile_clear() {
    az_core::prof::clear();
}

/// TEMPORARY (2026-10-06): proves which build is loaded.
#[pyfunction]
fn hot_sentinel() -> String {
    format!(
        "{} built {}",
        az_core::prof::SENTINEL,
        option_env!("HOTPROF_STAMP").unwrap_or("unstamped")
    )
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<chess::PyPosition>()?;
    m.add_class::<chess::PyChessSelfPlay>()?;
    m.add_class::<chess::PyChessSearch>()?;
    m.add_function(wrap_pyfunction!(chess::encode_batch, m)?)?;
    m.add_function(wrap_pyfunction!(chess::legal_mask_batch, m)?)?;
    m.add_function(wrap_pyfunction!(hot_profile, m)?)?;
    m.add_function(wrap_pyfunction!(hot_profile_clear, m)?)?;
    m.add_function(wrap_pyfunction!(hot_sentinel, m)?)?;

    m.add("POLICY_LEN", ChessPos::POLICY_LEN)?;
    m.add("OBS_SHAPE", ChessPos::OBS_SHAPE)?;
    m.add("OBS_PLANES", ChessPos::OBS_SHAPE.0)?;
    m.add("HISTORY", az_core::chess::HISTORY)?;

    connect4::register(m)?;
    Ok(())
}
