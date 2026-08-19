//! The Connect4 side of the bridge.
//!
//! Connect4 exists to prove the reinforcement-learning loop works before that
//! loop is pointed at chess. Everything here mirrors what the chess self-play
//! driver will expose, so a bug found at this scale is a bug fixed for both.

use az_core::connect4::{Connect4, HEIGHT, WIDTH};
use az_core::game::{obs_len, Game, Outcome, Player};
use az_core::mcts::{Config, Rng, Search, Status};
use az_core::selfplay::SelfPlay;
use ndarray::{Array1, Array2, Array4};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// `(obs, policy, z)` as handed to the training loop.
type TrainingArrays<'py> = (
    Bound<'py, PyArray4<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
);

const OBS: usize = 2 * HEIGHT * WIDTH;

fn outcome_str(o: Outcome) -> &'static str {
    match o {
        Outcome::Win => "win",
        Outcome::Loss => "loss",
        Outcome::Draw => "draw",
    }
}

// ---------------------------------------------------------------------------
// Position
// ---------------------------------------------------------------------------

#[pyclass(name = "Connect4", module = "chessbot_core", from_py_object)]
#[derive(Clone)]
pub struct PyConnect4 {
    pub(crate) inner: Connect4,
}

#[pymethods]
impl PyConnect4 {
    #[new]
    fn new() -> Self {
        Self {
            inner: Connect4::new(),
        }
    }

    /// Build a position by playing a sequence of 0-indexed columns.
    #[staticmethod]
    fn from_moves(cols: Vec<usize>) -> PyResult<Self> {
        Connect4::from_moves(&cols)
            .map(|inner| Self { inner })
            .ok_or_else(|| PyValueError::new_err("illegal move sequence"))
    }

    /// Playable columns.
    ///
    /// Returned as `u32`, not `u8`: PyO3 maps `Vec<u8>` to Python `bytes`, which
    /// then silently fails to work as a numpy index.
    fn legal_moves(&self) -> Vec<u32> {
        let mut v = Vec::new();
        self.inner.legal_moves(&mut v);
        v.into_iter().map(u32::from).collect()
    }

    /// Legal moves and terminal status from one call, as search uses it.
    fn expand(&self) -> (Vec<u32>, Option<&'static str>) {
        let mut v = Vec::new();
        let o = self.inner.expand(&mut v);
        (v.into_iter().map(u32::from).collect(), o.map(outcome_str))
    }

    fn outcome(&self) -> Option<&'static str> {
        self.inner.outcome().map(outcome_str)
    }

    fn play(&mut self, col: u8) -> PyResult<()> {
        if !self.inner.can_play(col as usize) {
            return Err(PyValueError::new_err(format!(
                "column {col} is not playable"
            )));
        }
        self.inner.play(col);
        Ok(())
    }

    fn after(&self, col: u8) -> PyResult<Self> {
        let mut next = self.clone();
        next.play(col)?;
        Ok(next)
    }

    /// True if dropping into `col` completes a four for the side to move.
    fn is_winning_move(&self, col: u8) -> bool {
        self.inner.is_winning_move(col as usize)
    }

    #[getter]
    fn plies(&self) -> u32 {
        self.inner.plies()
    }

    #[getter]
    fn turn(&self) -> &'static str {
        match self.inner.player_to_move() {
            Player::First => "first",
            Player::Second => "second",
        }
    }

    fn encode<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let mut buf = vec![0.0f32; OBS];
        self.inner.encode(&mut buf);
        ndarray::Array3::from_shape_vec((2, HEIGHT, WIDTH), buf)
            .map(|a| a.into_pyarray(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Board as text, row 5 (top) first, with `x` for the side to move.
    fn render(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("Connect4(plies={})", self.inner.plies())
    }
}

/// Encode many Connect4 positions into `(N, 2, 6, 7)`.
#[pyfunction]
pub(crate) fn connect4_encode_batch<'py>(
    py: Python<'py>,
    positions: Vec<PyConnect4>,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let n = positions.len();
    let buf = py.detach(|| {
        let mut buf = vec![0.0f32; n * OBS];
        for (i, p) in positions.iter().enumerate() {
            p.inner.encode(&mut buf[i * OBS..(i + 1) * OBS]);
        }
        buf
    });
    Array4::from_shape_vec((n, 2, HEIGHT, WIDTH), buf)
        .map(|a| a.into_pyarray(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

// ---------------------------------------------------------------------------
// Self-play driver
// ---------------------------------------------------------------------------

#[pyclass(name = "Connect4SelfPlay", module = "chessbot_core")]
pub struct PyConnect4SelfPlay {
    inner: SelfPlay<Connect4>,
    obs: Vec<f32>,
    batch: usize,
}

#[pymethods]
impl PyConnect4SelfPlay {
    #[new]
    #[pyo3(signature = (concurrency, total_games, sims=32, max_considered=7, max_plies=42, seed=0))]
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

    /// Step every live game to its next leaf.
    ///
    /// Returns `(N, 2, 6, 7)` observations, or `None` once every game is done.
    fn next_batch<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray4<f32>>>> {
        let obs = &mut self.obs;
        let inner = &mut self.inner;
        let n = py.detach(|| inner.next_batch(obs));
        self.batch = n;
        if n == 0 {
            return Ok(None);
        }
        Array4::from_shape_vec((n, 2, HEIGHT, WIDTH), self.obs.clone())
            .map(|a| Some(a.into_pyarray(py)))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Feed back one forward pass: `logits` is `(N, 7)`, `values` is `(N,)`.
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

    #[getter]
    fn evaluations(&self) -> u64 {
        self.inner.evaluations()
    }

    /// Drain finished games as training arrays.
    ///
    /// Returns `(obs, policy, z)` with shapes `(M, 2, 6, 7)`, `(M, 7)`, `(M,)`.
    /// The policy targets are the search's improved policy, not visit counts.
    fn take_training_data<'py>(&mut self, py: Python<'py>) -> PyResult<TrainingArrays<'py>> {
        let trajectories = self.inner.take_finished();
        let m: usize = trajectories.iter().map(|t| t.samples.len()).sum();

        let mut obs = vec![0.0f32; m * OBS];
        let mut policy = vec![0.0f32; m * WIDTH];
        let mut z = vec![0.0f32; m];

        let mut i = 0;
        for t in &trajectories {
            for s in &t.samples {
                s.pos.encode(&mut obs[i * OBS..(i + 1) * OBS]);
                for &(idx, p) in &s.policy {
                    policy[i * WIDTH + idx as usize] = p;
                }
                z[i] = s.z;
                i += 1;
            }
        }

        Ok((
            Array4::from_shape_vec((m, 2, HEIGHT, WIDTH), obs)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .into_pyarray(py),
            Array2::from_shape_vec((m, WIDTH), policy)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .into_pyarray(py),
            Array1::from_vec(z).into_pyarray(py),
        ))
    }

    /// Cumulative game statistics. Reading these does not consume anything,
    /// so it is safe to call alongside `take_training_data`.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (first, second, draws, mean_plies) = self.inner.stats();
        let d = PyDict::new(py);
        d.set_item("games", self.inner.games_completed())?;
        d.set_item("first_wins", first)?;
        d.set_item("second_wins", second)?;
        d.set_item("draws", draws)?;
        d.set_item("mean_plies", mean_plies)?;
        d.set_item("evaluations", self.inner.evaluations())?;
        Ok(d)
    }
}

// ---------------------------------------------------------------------------
// Batched search over arbitrary positions -- used by the arena and the
// tactical test suite.
// ---------------------------------------------------------------------------

#[pyclass(name = "Connect4Search", module = "chessbot_core")]
pub struct PyConnect4Search {
    searches: Vec<Search<Connect4>>,
    obs: Vec<f32>,
    batch_idx: Vec<usize>,
}

#[pymethods]
impl PyConnect4Search {
    /// Run one independent search per given position.
    #[new]
    #[pyo3(signature = (positions, sims=32, max_considered=7, seed=0))]
    fn new(positions: Vec<PyConnect4>, sims: u32, max_considered: usize, seed: u64) -> Self {
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
        let len = obs_len::<Connect4>();

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
        Array4::from_shape_vec((n, 2, HEIGHT, WIDTH), self.obs.clone())
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
            self.searches[i].apply(&logits[k * WIDTH..(k + 1) * WIDTH], values[k]);
        }
        Ok(())
    }

    /// Best move per position, in the order they were given.
    fn moves(&self) -> Vec<u32> {
        self.searches
            .iter()
            .map(|s| u32::from(s.result().0))
            .collect()
    }

    /// Improved-policy targets per position, shaped `(N, 7)`.
    fn policies<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let n = self.searches.len();
        let mut buf = vec![0.0f32; n * WIDTH];
        for (i, s) in self.searches.iter().enumerate() {
            let (_, target) = s.result();
            buf[i * WIDTH..(i + 1) * WIDTH].copy_from_slice(&target);
        }
        Array2::from_shape_vec((n, WIDTH), buf)
            .map(|a| a.into_pyarray(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Root value estimates after search, from each root mover's perspective.
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let v: Vec<f32> = self.searches.iter().map(|s| s.root_value()).collect();
        Array1::from_vec(v).into_pyarray(py)
    }
}

use numpy::{PyReadonlyArray1, PyReadonlyArray2};

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConnect4>()?;
    m.add_class::<PyConnect4SelfPlay>()?;
    m.add_class::<PyConnect4Search>()?;
    m.add_function(wrap_pyfunction!(connect4_encode_batch, m)?)?;
    m.add("CONNECT4_POLICY_LEN", Connect4::POLICY_LEN)?;
    m.add("CONNECT4_OBS_SHAPE", Connect4::OBS_SHAPE)?;
    Ok(())
}
