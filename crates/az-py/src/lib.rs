//! Python bindings for the AlphaZero core.
//!
//! # Where the boundary sits
//!
//! Python drives the training loop, but Rust owns every board and, once the
//! search lands, every tree. The boundary is crossed **once per neural-net
//! batch**, never once per node. That is the whole reason the hot path is in
//! Rust at all: a per-node crossing would cost more than the movegen it wraps.
//!
//! [`encode_batch`] is the primitive that shape rests on -- it turns many
//! positions into one contiguous `(N, C, 8, 8)` array in a single call, with the
//! GIL released for the actual work.
//!
//! # Move strings
//!
//! Moves cross the boundary as UCI strings (`e2e4`, `e7e8q`, `e1g1`). Castling is
//! the king's two-square move, matching python-chess, not shakmaty's internal
//! king-takes-rook form.

use az_core::chess::ChessPos;
use az_core::game::{Game, Outcome};
use ndarray::{Array3, Array4};
use numpy::{IntoPyArray, PyArray1, PyArray3, PyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use shakmaty::uci::UciMove;
use shakmaty::Move;

/// A chess position, with the history needed for repetition detection.
// `from_py_object` so batch calls can take `Vec<PyPosition>` directly; PyO3 0.29
// made that derive opt-in rather than automatic for Clone types.
#[pyclass(name = "Position", module = "chessbot_core", from_py_object)]
#[derive(Clone)]
pub struct PyPosition {
    inner: ChessPos,
}

impl PyPosition {
    fn resolve(&self, uci: &str) -> PyResult<Move> {
        let parsed: UciMove = uci
            .parse()
            .map_err(|_| PyValueError::new_err(format!("not a UCI move: {uci:?}")))?;
        parsed
            .to_move(self.inner.inner())
            .map_err(|_| PyValueError::new_err(format!("illegal move {uci:?} in {}", self.fen())))
    }
}

#[pymethods]
impl PyPosition {
    /// The standard starting position.
    #[new]
    fn new() -> Self {
        Self {
            inner: ChessPos::new(),
        }
    }

    #[staticmethod]
    fn from_fen(fen: &str) -> PyResult<Self> {
        ChessPos::from_fen(fen)
            .map(|inner| Self { inner })
            .map_err(PyValueError::new_err)
    }

    fn fen(&self) -> String {
        self.inner.fen()
    }

    /// `"white"` or `"black"`.
    #[getter]
    fn turn(&self) -> &'static str {
        match self.inner.player_to_move() {
            az_core::Player::First => "white",
            az_core::Player::Second => "black",
        }
    }

    /// Legal moves as UCI strings.
    fn legal_moves(&self) -> Vec<String> {
        let mut buf = Vec::new();
        self.inner.legal_moves(&mut buf);
        buf.into_iter()
            .map(|m| UciMove::from_standard(m).to_string())
            .collect()
    }

    /// `"win"`, `"loss"`, `"draw"` from the side to move's perspective, or
    /// `None` if the game is still going.
    fn outcome(&self) -> Option<&'static str> {
        self.inner.outcome().map(|o| match o {
            Outcome::Win => "win",
            Outcome::Loss => "loss",
            Outcome::Draw => "draw",
        })
    }

    /// Legal moves and terminal status from a single move generation.
    ///
    /// The pair that search needs at every node. A non-`None` outcome means the
    /// position is terminal *even when the move list is non-empty* -- fifty-move
    /// and repetition draws still have legal moves.
    fn expand(&self) -> (Vec<String>, Option<&'static str>) {
        let mut buf = Vec::new();
        let outcome = self.inner.expand(&mut buf);
        let moves = buf
            .into_iter()
            .map(|m| UciMove::from_standard(m).to_string())
            .collect();
        let outcome = outcome.map(|o| match o {
            Outcome::Win => "win",
            Outcome::Loss => "loss",
            Outcome::Draw => "draw",
        });
        (moves, outcome)
    }

    fn play(&mut self, uci: &str) -> PyResult<()> {
        let mv = self.resolve(uci)?;
        self.inner.play(mv);
        Ok(())
    }

    /// The position after `uci`, leaving this one untouched.
    fn after(&self, uci: &str) -> PyResult<Self> {
        let mv = self.resolve(uci)?;
        let mut next = self.inner.clone();
        next.play(mv);
        Ok(Self { inner: next })
    }

    /// Index of `uci` in the 4672-long flat policy vector.
    fn policy_index(&self, uci: &str) -> PyResult<usize> {
        let mv = self.resolve(uci)?;
        Ok(self.inner.policy_index(mv))
    }

    /// Boolean mask over the policy vector, true where a move is legal.
    ///
    /// Multiply or `where` this against the policy head's logits before
    /// softmaxing; the network has no idea which of the 4672 slots are real.
    fn legal_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        let mut mask = vec![false; ChessPos::POLICY_LEN];
        let mut buf = Vec::new();
        self.inner.legal_moves(&mut buf);
        for m in buf {
            mask[self.inner.policy_index(m)] = true;
        }
        mask.into_pyarray(py)
    }

    /// Observation planes for this position, shaped `(C, 8, 8)`.
    fn encode<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let (c, h, w) = ChessPos::OBS_SHAPE;
        let mut buf = vec![0.0f32; c * h * w];
        self.inner.encode(&mut buf);
        Array3::from_shape_vec((c, h, w), buf)
            .map(|a| a.into_pyarray(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// How many times this position has occurred, including now.
    fn repetition_count(&self) -> usize {
        self.inner.repetition_count()
    }

    fn __repr__(&self) -> String {
        format!("Position({:?})", self.fen())
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }
}

/// Encode many positions into one contiguous `(N, C, 8, 8)` array.
///
/// This is the shape of every neural-net call in self-play: one crossing, one
/// allocation, one array. The encoding itself runs with the GIL released, so the
/// only serialised part is the argument marshalling.
#[pyfunction]
fn encode_batch<'py>(
    py: Python<'py>,
    positions: Vec<PyPosition>,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let (c, h, w) = ChessPos::OBS_SHAPE;
    let stride = c * h * w;
    let n = positions.len();

    let buf = py.detach(|| {
        let mut buf = vec![0.0f32; n * stride];
        for (i, p) in positions.iter().enumerate() {
            p.inner.encode(&mut buf[i * stride..(i + 1) * stride]);
        }
        buf
    });

    Array4::from_shape_vec((n, c, h, w), buf)
        .map(|a| a.into_pyarray(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Legal-move masks for many positions, shaped `(N, POLICY_LEN)`.
#[pyfunction]
fn legal_mask_batch<'py>(
    py: Python<'py>,
    positions: Vec<PyPosition>,
) -> PyResult<Bound<'py, numpy::PyArray2<bool>>> {
    let n = positions.len();
    let len = ChessPos::POLICY_LEN;

    let buf = py.detach(|| {
        let mut buf = vec![false; n * len];
        let mut moves = Vec::new();
        for (i, p) in positions.iter().enumerate() {
            p.inner.legal_moves(&mut moves);
            for &m in &moves {
                buf[i * len + p.inner.policy_index(m)] = true;
            }
        }
        buf
    });

    ndarray::Array2::from_shape_vec((n, len), buf)
        .map(|a| a.into_pyarray(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pymodule]
fn chessbot_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPosition>()?;
    m.add_function(wrap_pyfunction!(encode_batch, m)?)?;
    m.add_function(wrap_pyfunction!(legal_mask_batch, m)?)?;

    m.add("POLICY_LEN", ChessPos::POLICY_LEN)?;
    m.add("OBS_SHAPE", ChessPos::OBS_SHAPE)?;
    m.add("OBS_PLANES", ChessPos::OBS_SHAPE.0)?;
    m.add("HISTORY", az_core::chess::HISTORY)?;
    Ok(())
}
