//! The chess side of the bridge: positions, encodings and batch primitives.

use az_core::chess::ChessPos;
use az_core::game::obs_len;
use az_core::game::{Game, Outcome};
use az_core::mcts::{Config, Rng, Search, Status};
use az_core::selfplay::SelfPlay;
use ndarray::{Array1, Array2, Array3, Array4};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyReadonlyArray1, PyReadonlyArray2,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
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
pub(crate) fn encode_batch<'py>(
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
pub(crate) fn legal_mask_batch<'py>(
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

// ---------------------------------------------------------------------------
// Self-play driver
// ---------------------------------------------------------------------------

const OBS: usize = 119 * 64;

fn outcome_str(o: az_core::game::Outcome) -> &'static str {
    use az_core::game::Outcome::*;
    match o {
        Win => "win",
        Loss => "loss",
        Draw => "draw",
    }
}

/// `(obs, policy, z, value_mask)` as handed to the training loop.
///
/// `value_mask` is 0 for positions from games the ply limit cut short. Those
/// games have no known result, so `z` there is a fabrication and the value loss
/// must skip them.
type TrainingArrays<'py> = (
    Bound<'py, PyArray4<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
);

/// Batched chess self-play.
///
/// Identical in shape to the Connect4 driver, because it is the same driver --
/// `az_core::selfplay` is generic over the `Game` trait. Everything the Connect4
/// run proved about the loop carries over unchanged.
#[pyclass(name = "ChessSelfPlay", module = "chessbot_core")]
pub struct PyChessSelfPlay {
    inner: SelfPlay<ChessPos>,
    obs: Vec<f32>,
    batch: usize,
}

#[pymethods]
impl PyChessSelfPlay {
    #[new]
    #[pyo3(signature = (concurrency, total_games, sims=32, max_considered=16, max_plies=240, seed=0))]
    fn new(
        concurrency: usize,
        total_games: usize,
        sims: u32,
        max_considered: usize,
        max_plies: usize,
        seed: u64,
    ) -> Self {
        let cfg = Config {
            sims,
            max_considered,
            ..Config::default()
        };
        Self {
            inner: SelfPlay::new(concurrency, total_games, cfg, max_plies, seed),
            obs: Vec::new(),
            batch: 0,
        }
    }

    /// Step every live game to its next leaf; `None` once all games are done.
    fn next_batch<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray4<f32>>>> {
        let obs = &mut self.obs;
        let inner = &mut self.inner;
        let n = py.detach(|| inner.next_batch(obs));
        self.batch = n;
        if n == 0 {
            return Ok(None);
        }
        Array4::from_shape_vec((n, 119, 8, 8), self.obs.clone())
            .map(|a| Some(a.into_pyarray(py)))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Feed back one forward pass: `logits` is `(N, 4672)`, `values` is `(N,)`.
    fn submit(
        &mut self,
        logits: PyReadonlyArray2<f32>,
        values: PyReadonlyArray1<f32>,
    ) -> PyResult<()> {
        let logits = logits.as_slice()?;
        let values = values.as_slice()?;
        if values.len() != self.batch {
            return Err(PyValueError::new_err(format!(
                "expected {} values, got {}",
                self.batch,
                values.len()
            )));
        }
        self.inner.submit(logits, values);
        Ok(())
    }

    fn is_done(&self) -> bool {
        self.inner.is_done()
    }

    #[getter]
    fn games_completed(&self) -> usize {
        self.inner.games_completed()
    }

    /// Drain finished games as `(obs, policy, z, value_mask)`.
    ///
    /// The policy target is dense over all 4672 slots, which is 18 KB per position
    /// -- large, but it is the shape the loss wants and the buffer is drained every
    /// generation.
    ///
    /// `value_mask` is zero for games stopped by the ply limit. Such a game has no
    /// result; calling it a draw would teach the value head an outcome that never
    /// happened.
    fn take_training_data<'py>(&mut self, py: Python<'py>) -> PyResult<TrainingArrays<'py>> {
        let trajectories = self.inner.take_finished();
        let m: usize = trajectories.iter().map(|t| t.samples.len()).sum();

        let mut obs = vec![0.0f32; m * OBS];
        let mut policy = vec![0.0f32; m * 4672];
        let mut z = vec![0.0f32; m];
        let mut mask = vec![0.0f32; m];

        let mut i = 0;
        for t in &trajectories {
            let known = if t.decided { 1.0 } else { 0.0 };
            for s in &t.samples {
                s.pos.encode(&mut obs[i * OBS..(i + 1) * OBS]);
                for &(idx, p) in &s.policy {
                    policy[i * 4672 + idx as usize] = p;
                }
                z[i] = s.z;
                mask[i] = known;
                i += 1;
            }
        }

        Ok((
            Array4::from_shape_vec((m, 119, 8, 8), obs)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .into_pyarray(py),
            Array2::from_shape_vec((m, 4672), policy)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .into_pyarray(py),
            Array1::from_vec(z).into_pyarray(py),
            Array1::from_vec(mask).into_pyarray(py),
        ))
    }

    /// Cumulative statistics. Reading these consumes nothing.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (white, black, draws, mean_plies) = self.inner.stats();
        let d = PyDict::new(py);
        d.set_item("games", self.inner.games_completed())?;
        d.set_item("white_wins", white)?;
        d.set_item("black_wins", black)?;
        d.set_item("draws", draws)?;
        d.set_item("mean_plies", mean_plies)?;
        d.set_item("truncated", self.inner.truncated())?;
        d.set_item("evaluations", self.inner.evaluations())?;
        Ok(d)
    }
}

// ---------------------------------------------------------------------------
// Batched search over arbitrary positions
// ---------------------------------------------------------------------------

#[pyclass(name = "ChessSearch", module = "chessbot_core")]
pub struct PyChessSearch {
    searches: Vec<Search<ChessPos>>,
    obs: Vec<f32>,
    batch_idx: Vec<usize>,
}

#[pymethods]
impl PyChessSearch {
    /// One independent search per position, all sharing the network batches.
    #[new]
    #[pyo3(signature = (positions, sims=32, max_considered=16, seed=0))]
    fn new(positions: Vec<PyPosition>, sims: u32, max_considered: usize, seed: u64) -> Self {
        let cfg = Config {
            sims,
            max_considered,
            ..Config::default()
        };
        let mut rng = Rng::new(seed);
        let searches = positions
            .into_iter()
            .map(|p| Search::new(p.inner, cfg, &mut rng))
            .collect();
        Self {
            searches,
            obs: Vec::new(),
            batch_idx: Vec::new(),
        }
    }

    fn next_batch<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray4<f32>>>> {
        self.obs.clear();
        self.batch_idx.clear();
        let len = obs_len::<ChessPos>();

        for i in 0..self.searches.len() {
            if self.searches[i].prepare() == Status::NeedsEval {
                let start = self.obs.len();
                self.obs.resize(start + len, 0.0);
                self.searches[i].pending().encode(&mut self.obs[start..]);
                self.batch_idx.push(i);
            }
        }

        let n = self.batch_idx.len();
        if n == 0 {
            return Ok(None);
        }
        Array4::from_shape_vec((n, 119, 8, 8), self.obs.clone())
            .map(|a| Some(a.into_pyarray(py)))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn submit(
        &mut self,
        logits: PyReadonlyArray2<f32>,
        values: PyReadonlyArray1<f32>,
    ) -> PyResult<()> {
        let logits = logits.as_slice()?;
        let values = values.as_slice()?;
        for (k, &i) in self.batch_idx.iter().enumerate() {
            self.searches[i].apply(&logits[k * 4672..(k + 1) * 4672], values[k]);
        }
        Ok(())
    }

    /// Best move per position as UCI, in the order they were given.
    fn moves(&self) -> Vec<String> {
        self.searches
            .iter()
            .map(|s| UciMove::from_standard(s.result().0).to_string())
            .collect()
    }

    /// Root value estimates after search, from each root mover's perspective.
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let v: Vec<f32> = self.searches.iter().map(|s| s.root_value()).collect();
        Array1::from_vec(v).into_pyarray(py)
    }

    /// Results the rules guarantee at each root: `"win"`, `"loss"`, `"draw"`, or
    /// `None` where the search proved nothing.
    ///
    /// A UCI engine should report these as mate scores rather than as an
    /// evaluation -- they are certainties, not estimates.
    fn proven(&self) -> Vec<Option<&'static str>> {
        self.searches
            .iter()
            .map(|s| s.proven().map(outcome_str))
            .collect()
    }

    /// Simulations actually spent, per position.
    fn simulations(&self) -> Vec<u32> {
        self.searches.iter().map(|s| s.simulations()).collect()
    }

    /// Tree nodes allocated, per position -- the closest thing to a node count
    /// that UCI's `info nodes` expects.
    fn nodes(&self) -> Vec<usize> {
        self.searches.iter().map(|s| s.nodes_allocated()).collect()
    }
}
