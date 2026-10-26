//! Batched self-play driver.
//!
//! Runs many games concurrently and pools their leaf evaluations into a single
//! network call. This is not an optimisation, it is the whole design: a network
//! forward at batch 1 costs the same wall-clock as one at batch 512 on Metal, so
//! a one-game-at-a-time search leaves the GPU roughly 98% idle. See
//! `bench/results/2026-08-16-bridge-throughput.md`.
//!
//! Because Gumbel search runs at a small simulation budget, each tree only needs
//! **one** leaf evaluated per pass. That removes the need for virtual loss
//! entirely -- concurrency comes from running many games, not from forcing one
//! tree to hand over several leaves at once. Batch size is simply the number of
//! live games.
//!
//! ```text
//! loop {
//!     let n = sp.next_batch(&mut obs);      // Rust: step every game to its next leaf
//!     if n == 0 { break }
//!     let (logits, values) = net(obs);      // Python: one forward
//!     sp.submit(&logits, &values);          // Rust: expand, back up, advance
//! }
//! ```
//!
//! # Threading
//!
//! `threads` splits the slots into that many contiguous shards and steps them in
//! parallel. A slot is a whole game, so this is parallelism at *game*
//! granularity: no two threads ever touch the same tree, the same trajectory or
//! the same row of the observation buffer.
//!
//! That distinction is the entire lesson of the first attempt. On 2026-10-06 the
//! same loops were handed to `rayon::par_iter_mut` over individual slots and
//! measured *slower* -- 4.8 s to 5.4 s at 4 threads (`86b661a`, reverted in
//! `2131d58`). A call carries only ~5 ms of work, so per-slot task dispatch and
//! work-stealing cost more than the work they hand out. Here the pool is sized
//! once, the shards are fixed, and a call dispatches `threads` tasks rather than
//! `concurrency` of them.
//!
//! `threads = 1` takes a path with no pool and no rayon call at all, so the
//! sequential driver that has shipped since August is unchanged rather than
//! merely equivalent.

use crate::game::{obs_len, Game, Outcome, Player};
use crate::mcts::{Config, Rng, Search, Status};
use crate::{prof_count, prof_time};

/// One training example: a position, the search's improved policy, and the
/// eventual game result from that position's mover's point of view.
/// Playout cap randomization (KataGo, Wu 2019), adapted to a Gumbel search.
///
/// Value targets want many games, so they want cheap moves. Policy targets want
/// a search worth imitating, so they want expensive ones. AlphaZero resolves
/// this by paying the expensive price everywhere. PCR instead plays most moves
/// cheaply and takes policy targets only from the rare expensive ones.
///
/// The Gumbel-specific part is choosing `full.sims`. Sequential halving spends
/// `sims / (phases * m)` visits per contender per phase, floored to one -- so at
/// `sims == m` the whole budget goes on a single pass and **halving never runs**.
/// At m = 32 that is exactly our default, and the cheap path leans into it
/// (narrow shortlist, one visit each) while the full path is sized so halving
/// actually executes.
#[derive(Debug, Clone, Copy)]
pub struct Pcr {
    /// Probability that a move gets the full budget and yields a policy target.
    pub prob: f32,
    /// Search settings for the cheap majority of moves.
    pub fast: Config,
}

pub struct Sample<G: Game> {
    pub pos: G,
    /// Sparse `(policy_index, probability)`. Dense would be 4672 floats per ply
    /// in chess, which at hundreds of concurrent games is gigabytes for nothing.
    pub policy: Vec<(u32, f32)>,
    /// Game result in [-1, 1]. Filled in once the game ends.
    pub z: f32,
    /// Search's value estimate at this position, for diagnostics.
    pub root_value: f32,
    /// 1.0 when this position was searched at the full budget and its policy
    /// target is worth training on; 0.0 when it came from a cheap move.
    ///
    /// Cheap positions still carry a value target -- the game result is just as
    /// true for them -- which is the entire point of the split.
    pub policy_mask: f32,
}

pub struct Trajectory<G: Game> {
    pub samples: Vec<Sample<G>>,
    /// Winner, or `None` for a draw.
    pub winner: Option<Player>,
    pub plies: usize,
    /// False if the game was cut short by the ply limit rather than decided by
    /// the rules.
    ///
    /// The distinction matters for training: a truncated game has *no known
    /// result*. Recording it as a draw invents a label, and the value head then
    /// learns that invention.
    pub decided: bool,
}

/// Everything one game owns, including its own share of the bookkeeping.
///
/// The tallies and the finished-game list live here rather than on `SelfPlay`
/// because a thread stepping this slot must be able to end a game without
/// touching anything another thread can see. They are summed on read.
struct Slot<G: Game> {
    pos: G,
    search: Search<G>,
    samples: Vec<Sample<G>>,
    live: bool,
    /// Whether the search currently in flight was given the full budget.
    full: bool,
    /// Did this slot put a row in the batch now being built?
    produced: bool,
    /// Games this slot still has to start after the one in flight.
    ///
    /// Fixed at construction rather than drawn from a shared pool. Whichever
    /// slot happened to finish first used to take the next game, which made the
    /// game-to-slot assignment depend on completion order -- harmless while the
    /// driver was sequential, and a silent source of irreproducibility the
    /// moment it is not. A fixed quota makes a slot's whole sequence of games a
    /// function of the seed and the slot index alone, so a run reproduces for
    /// any `threads`.
    ///
    /// In the training configuration `concurrency == games`, so every quota is
    /// zero and this changes nothing at all.
    quota: usize,
    /// This slot's own random stream, derived from the master seed and the slot
    /// index.
    ///
    /// A single shared generator would make the games depend on the *order* in
    /// which slots consume it, which is fine while the driver is sequential and
    /// fatal once it is not: thread scheduling would decide the Gumbel noise and
    /// a seed would no longer reproduce a run. Per-slot streams make slot order
    /// irrelevant, which is what lets the loops below be parallel at all.
    rng: Rng,

    finished: Vec<Trajectory<G>>,
    games_completed: usize,
    first_wins: usize,
    second_wins: usize,
    draws: usize,
    total_plies: usize,
    truncated: usize,
}

pub struct SelfPlay<G: Game> {
    cfg: Config,
    /// `None` reproduces the pre-PCR behaviour exactly: every move full budget,
    /// every position a policy target.
    pcr: Option<Pcr>,
    max_plies: usize,
    slots: Vec<Slot<G>>,

    /// Slot index for each row of the batch produced by `next_batch`.
    batch_slots: Vec<usize>,

    evaluations: u64,

    /// Driver threads. 1 means the sequential path.
    threads: usize,
    #[cfg(feature = "parallel")]
    pool: Option<rayon::ThreadPool>,
}

/// Does this move get the full budget? A fresh draw per move.
fn roll(rng: &mut Rng, pcr: Option<Pcr>) -> bool {
    match pcr {
        None => true,
        Some(p) => (rng.next_u64() >> 11) as f32 / (1u64 << 53) as f32 <= p.prob,
    }
}

/// The search config a move of this kind runs under.
fn budget(cfg: Config, pcr: Option<Pcr>, full: bool) -> Config {
    match pcr {
        Some(p) if !full => p.fast,
        _ => cfg,
    }
}

impl<G: Game> Slot<G> {
    /// Step this one game to its next leaf, encoding it into `row`.
    ///
    /// `row` is exactly this slot's `obs_len` floats and nobody else's, which is
    /// what makes the whole function safe to run on its own thread.
    fn step(&mut self, row: &mut [f32], cfg: Config, pcr: Option<Pcr>, max_plies: usize) {
        self.produced = false;
        if !self.live {
            return;
        }
        loop {
            prof_count!("n_prepare", 1);
            let st = prof_time!("ns_prepare", self.search.prepare());
            match st {
                Status::NeedsEval => {
                    prof_time!("ns_encode", self.search.pending().encode(row));
                    self.produced = true;
                    return;
                }
                Status::Complete => {
                    // Loop rather than return even when the game ends: the slot
                    // has just been refilled with a fresh game, and that game's
                    // root evaluation belongs in *this* batch.
                    prof_count!("n_commit", 1);
                    prof_time!("ns_commit", self.commit_move(cfg, pcr, max_plies));
                    if !self.live {
                        return;
                    }
                }
            }
        }
    }

    /// Play the move the search settled on.
    fn commit_move(&mut self, cfg: Config, pcr: Option<Pcr>, max_plies: usize) {
        let (mv, target) = prof_time!("ns_result", self.search.result());
        let root_value = prof_time!("ns_root_value", self.search.root_value());

        let policy: Vec<(u32, f32)> = target
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(idx, &p)| (idx as u32, p))
            .collect();

        let pos = self.pos.clone();
        let policy_mask = if self.full { 1.0 } else { 0.0 };
        self.samples.push(Sample {
            pos,
            policy,
            z: 0.0,
            root_value,
            policy_mask,
        });

        self.pos.play(mv);

        let mut scratch = Vec::new();
        let outcome = self.pos.expand(&mut scratch);
        if outcome.is_some() || self.samples.len() >= max_plies {
            self.finish_game(outcome, cfg, pcr);
            return;
        }

        let next = self.pos.clone();
        self.full = roll(&mut self.rng, pcr);
        prof_count!("n_search_new", 1);
        self.search = prof_time!(
            "ns_search_new",
            Search::new(next, budget(cfg, pcr, self.full), &mut self.rng)
        );
    }

    fn finish_game(&mut self, outcome: Option<Outcome>, cfg: Config, pcr: Option<Pcr>) {
        // `None` here means the ply limit stopped the game, not that it was drawn.
        let decided = outcome.is_some();
        let final_mover = self.pos.player_to_move();
        let winner = match outcome {
            Some(Outcome::Loss) => Some(final_mover.other()),
            Some(Outcome::Win) => Some(final_mover),
            // A draw, or a game cut short by the ply limit.
            _ => None,
        };

        let mut samples = std::mem::take(&mut self.samples);
        for s in &mut samples {
            s.z = match winner {
                None => 0.0,
                Some(w) if w == s.pos.player_to_move() => 1.0,
                Some(_) => -1.0,
            };
        }

        let plies = samples.len();
        match winner {
            Some(Player::First) => self.first_wins += 1,
            Some(Player::Second) => self.second_wins += 1,
            None => self.draws += 1,
        }
        self.total_plies += plies;
        if !decided {
            self.truncated += 1;
        }
        self.finished.push(Trajectory {
            samples,
            winner,
            plies,
            decided,
        });
        self.games_completed += 1;

        // Refill from this slot's own quota so the next batch stays full.
        if self.quota > 0 {
            self.quota -= 1;
            self.pos = G::initial();
            self.full = roll(&mut self.rng, pcr);
            prof_count!("n_search_new", 1);
            self.search = prof_time!(
                "ns_search_new",
                Search::new(G::initial(), budget(cfg, pcr, self.full), &mut self.rng)
            );
            self.samples.clear();
        } else {
            self.live = false;
        }
    }
}

impl<G: Game> SelfPlay<G> {
    /// `concurrency` games in flight at once; `total_games` played in all.
    ///
    /// Finished games are replaced immediately so the batch stays full: letting
    /// it drain would spend the tail of every run at a batch size where the GPU
    /// is idle.
    pub fn new(
        concurrency: usize,
        total_games: usize,
        cfg: Config,
        max_plies: usize,
        seed: u64,
    ) -> Self {
        Self::with_pcr(concurrency, total_games, cfg, max_plies, seed, None)
    }

    /// As [`SelfPlay::new`], with playout cap randomization.
    pub fn with_pcr(
        concurrency: usize,
        total_games: usize,
        cfg: Config,
        max_plies: usize,
        seed: u64,
        pcr: Option<Pcr>,
    ) -> Self {
        let concurrency = concurrency.max(1).min(total_games.max(1));
        // Slot k draws from its own stream. The odd multiplier keeps nearby
        // seeds from producing correlated streams on an xorshift generator.
        let slots = (0..concurrency)
            .map(|k| {
                let mut srng = Rng::new(seed ^ (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let full = roll(&mut srng, pcr);
                // Deal the games out round-robin: slot k plays game indices
                // k, k + concurrency, k + 2*concurrency, ... The one in flight
                // is the first of them, so the quota is the share minus one.
                let share = total_games / concurrency + (k < total_games % concurrency) as usize;
                Slot {
                    pos: G::initial(),
                    search: Search::new(G::initial(), budget(cfg, pcr, full), &mut srng),
                    samples: Vec::new(),
                    live: share > 0,
                    full,
                    produced: false,
                    quota: share.saturating_sub(1),
                    rng: srng,
                    finished: Vec::new(),
                    games_completed: 0,
                    first_wins: 0,
                    second_wins: 0,
                    draws: 0,
                    total_plies: 0,
                    truncated: 0,
                }
            })
            .collect();

        Self {
            cfg,
            pcr,
            max_plies,
            slots,
            batch_slots: Vec::new(),
            evaluations: 0,
            threads: 1,
            #[cfg(feature = "parallel")]
            pool: None,
        }
    }

    /// Step the games on `threads` threads instead of one.
    ///
    /// Clamped to the slot count: more threads than shards would leave some with
    /// nothing to do while still paying to wake them. Returns the number
    /// actually used, which is what a benchmark should report.
    ///
    /// This does not change what the run produces. Every slot owns its random
    /// stream, its game quota and its own row of the batch, so the trajectories
    /// are a function of the seed alone -- `driver_threads_do_not_change_the_games`
    /// asserts exactly that.
    pub fn set_threads(&mut self, threads: usize) -> usize {
        self.threads = threads.max(1).min(self.slots.len().max(1));
        #[cfg(feature = "parallel")]
        {
            self.pool = (self.threads > 1).then(|| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(self.threads)
                    // Named so that a stuck driver is identifiable in a stack
                    // dump, and its own pool so it never competes with torch's
                    // intra-op threads for rayon's global one.
                    .thread_name(|i| format!("az-driver-{i}"))
                    .build()
                    .expect("driver thread pool")
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.threads = 1;
        }
        self.threads
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Step every live game to its next leaf and encode them all.
    ///
    /// Returns the batch size `n`; zero means every game has finished. The
    /// first `n * obs_len` floats of `out` are the batch, row-major
    /// `(N, C, H, W)`; anything past that is stale and must be ignored.
    ///
    /// `out` is grown to the full slot count and then left alone. Clearing and
    /// re-`resize`ing it each call zero-filled every observation twice, once
    /// here and once in `encode`, which is 30 KB of pointless memset per
    /// evaluation.
    pub fn next_batch(&mut self, out: &mut Vec<f32>) -> usize {
        let __nb = std::time::Instant::now();
        prof_count!("n_next_batch", 1);
        let obs = obs_len::<G>();
        let needed = self.slots.len() * obs;
        if out.len() < needed {
            out.resize(needed, 0.0);
        }
        self.batch_slots.clear();

        // Each slot encodes into *its own* row, out[i*obs ..][..obs], so the
        // shards below write to disjoint memory and need no coordination. The
        // batch is compacted afterwards, which costs nothing until games start
        // running out: while every slot is live, every slot produces a row and
        // row i is already in the right place.
        let (cfg, pcr, max_plies) = (self.cfg, self.pcr, self.max_plies);
        let rows = &mut out[..needed];

        #[cfg(feature = "parallel")]
        if let Some(pool) = &self.pool {
            use rayon::prelude::*;
            let per = self.slots.len().div_ceil(self.threads);
            let slots = &mut self.slots;
            pool.install(|| {
                slots
                    .par_chunks_mut(per)
                    .zip(rows.par_chunks_mut(per * obs))
                    .for_each(|(shard, buf)| {
                        for (k, slot) in shard.iter_mut().enumerate() {
                            slot.step(&mut buf[k * obs..(k + 1) * obs], cfg, pcr, max_plies);
                        }
                    });
            });
        } else {
            for (i, slot) in self.slots.iter_mut().enumerate() {
                slot.step(&mut rows[i * obs..(i + 1) * obs], cfg, pcr, max_plies);
            }
        }
        #[cfg(not(feature = "parallel"))]
        for (i, slot) in self.slots.iter_mut().enumerate() {
            slot.step(&mut rows[i * obs..(i + 1) * obs], cfg, pcr, max_plies);
        }

        // Close the gaps left by slots that have run out of games. A copy only
        // happens once a row's index has drifted from its slot's, so the steady
        // state moves nothing.
        for i in 0..self.slots.len() {
            if !self.slots[i].produced {
                continue;
            }
            let dst = self.batch_slots.len();
            if dst != i {
                rows.copy_within(i * obs..(i + 1) * obs, dst * obs);
            }
            self.batch_slots.push(i);
        }

        self.evaluations += self.batch_slots.len() as u64;
        prof_count!("n_evals", self.batch_slots.len());
        prof_count!("ns_next_batch", __nb.elapsed().as_nanos());
        self.batch_slots.len()
    }

    /// Feed back one forward pass.
    ///
    /// `logits` is `n * POLICY_LEN`, `values` is `n`, both in the order
    /// [`SelfPlay::next_batch`] produced.
    pub fn submit(&mut self, logits: &[f32], values: &[f32]) {
        let __sb = std::time::Instant::now();
        let n = self.batch_slots.len();
        assert_eq!(values.len(), n, "expected {n} values");
        assert_eq!(logits.len(), n * G::POLICY_LEN, "expected {n} policy rows");

        // Cheap next to the descent in `next_batch` -- one `apply` per row, no
        // tree walk -- so it stays sequential and the pool is not woken twice
        // per batch.
        for (k, &slot) in self.batch_slots.iter().enumerate() {
            let row = &logits[k * G::POLICY_LEN..(k + 1) * G::POLICY_LEN];
            prof_time!("ns_apply", self.slots[slot].search.apply(row, values[k]));
        }
        prof_count!("ns_submit", __sb.elapsed().as_nanos());
    }

    pub fn is_done(&self) -> bool {
        self.slots.iter().all(|s| !s.live)
    }

    pub fn games_completed(&self) -> usize {
        self.slots.iter().map(|s| s.games_completed).sum()
    }

    /// Total leaf evaluations requested so far.
    pub fn evaluations(&self) -> u64 {
        self.evaluations
    }

    /// `(first_wins, second_wins, draws, mean_plies)` over every game so far.
    ///
    /// Reading this does not consume anything.
    pub fn stats(&self) -> (usize, usize, usize, f64) {
        let mut first = 0;
        let mut second = 0;
        let mut draws = 0;
        let mut plies = 0;
        for s in &self.slots {
            first += s.first_wins;
            second += s.second_wins;
            draws += s.draws;
            plies += s.total_plies;
        }
        let n = self.games_completed().max(1);
        (first, second, draws, plies as f64 / n as f64)
    }

    /// How many finished games ran out of plies instead of reaching a result.
    pub fn truncated(&self) -> usize {
        self.slots.iter().map(|s| s.truncated).sum()
    }

    /// Drain the finished games.
    ///
    /// Grouped by slot rather than interleaved in completion order, because the
    /// slots are what the threads own. The training buffer is shuffled before
    /// use, so order carries no information -- but an A/B that compares two
    /// drivers should compare the *set* of games, not the sequence.
    pub fn take_finished(&mut self) -> Vec<Trajectory<G>> {
        let mut out = Vec::new();
        for s in &mut self.slots {
            out.append(&mut s.finished);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connect4::Connect4;

    /// As `run_flat`, with playout cap randomization.
    fn run_pcr(pcr: Option<Pcr>) -> Vec<Trajectory<Connect4>> {
        run_pcr_on(pcr, 1)
    }

    fn run_pcr_on(pcr: Option<Pcr>, threads: usize) -> Vec<Trajectory<Connect4>> {
        let cfg = Config { sims: 64, max_considered: 7, ..Config::default() };
        let mut sp = SelfPlay::<Connect4>::with_pcr(8, 24, cfg, 42, 1234, pcr);
        sp.set_threads(threads);
        let mut obs = Vec::new();
        let mut guard = 0;
        loop {
            let n = sp.next_batch(&mut obs);
            if n == 0 {
                break;
            }
            let logits = vec![0.0f32; n * Connect4::POLICY_LEN];
            let values = vec![0.0f32; n];
            sp.submit(&logits, &values);
            guard += 1;
            assert!(guard < 200_000, "driver failed to terminate");
        }
        sp.take_finished()
    }

    /// Slot order must not reach the games.
    ///
    /// Each slot owns a stream derived from the master seed and its index, so a
    /// run is reproducible no matter what order slots consume randomness in.
    /// That was introduced for a parallel driver which measured slower and was
    /// reverted; the property is kept because it is the prerequisite for ever
    /// retrying, and because a single shared generator is a trap either way.
    #[test]
    fn each_slot_draws_from_its_own_stream() {
        let a = run_pcr(None);
        let b = run_pcr(None);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.plies, y.plies);
            assert_eq!(x.winner, y.winner);
            for (p, q) in x.samples.iter().zip(y.samples.iter()) {
                assert_eq!(p.policy, q.policy);
            }
        }
    }

    /// Without PCR every position is a policy target, exactly as before.
    #[test]
    fn no_pcr_marks_every_position_as_a_policy_target() {
        let games = run_pcr(None);
        assert!(!games.is_empty());
        for g in &games {
            for s in &g.samples {
                assert_eq!(s.policy_mask, 1.0);
            }
        }
    }

    /// With PCR only the rare full-budget moves carry a policy target, while
    /// every position still carries a value target -- which is the whole point.
    #[test]
    fn pcr_marks_roughly_the_requested_fraction() {
        let fast = Config { sims: 8, max_considered: 4, ..Config::default() };
        let games = run_pcr(Some(Pcr { prob: 0.25, fast }));
        let (mut full, mut total) = (0usize, 0usize);
        for g in &games {
            for s in &g.samples {
                assert!(s.policy_mask == 0.0 || s.policy_mask == 1.0);
                full += (s.policy_mask == 1.0) as usize;
                total += 1;
            }
        }
        assert!(total > 200, "too few samples to judge: {total}");
        let frac = full as f64 / total as f64;
        assert!(
            (0.15..0.35).contains(&frac),
            "asked for 0.25 of moves at full budget, got {frac:.3}"
        );
    }

    /// Uniform policy, zero value: exercises the driver, not learning.
    fn run_flat(concurrency: usize, total: usize, sims: u32) -> Vec<Trajectory<Connect4>> {
        run_flat_on(concurrency, total, sims, 1)
    }

    fn run_flat_on(
        concurrency: usize,
        total: usize,
        sims: u32,
        threads: usize,
    ) -> Vec<Trajectory<Connect4>> {
        let cfg = Config {
            sims,
            max_considered: 7,
            ..Config::default()
        };
        let mut sp = SelfPlay::<Connect4>::new(concurrency, total, cfg, 42, 1234);
        sp.set_threads(threads);
        let mut obs = Vec::new();
        let mut guard = 0;

        loop {
            let n = sp.next_batch(&mut obs);
            if n == 0 {
                break;
            }
            let logits = vec![0.0f32; n * Connect4::POLICY_LEN];
            let values = vec![0.0f32; n];
            sp.submit(&logits, &values);

            guard += 1;
            assert!(guard < 1_000_000, "driver failed to terminate");
        }
        sp.take_finished()
    }


    /// Compare two runs game for game, not just in aggregate.
    fn same_games(a: &[Trajectory<Connect4>], b: &[Trajectory<Connect4>], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: different game counts");
        for (n, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x.plies, y.plies, "{what}: game {n} length");
            assert_eq!(x.winner, y.winner, "{what}: game {n} winner");
            assert_eq!(x.decided, y.decided, "{what}: game {n} decided");
            for (k, (p, q)) in x.samples.iter().zip(&y.samples).enumerate() {
                assert_eq!(p.policy, q.policy, "{what}: game {n} ply {k} policy");
                assert_eq!(p.z, q.z, "{what}: game {n} ply {k} z");
                assert_eq!(p.policy_mask, q.policy_mask, "{what}: game {n} ply {k} mask");
            }
        }
    }

    /// The property the whole threaded driver rests on: a seed reproduces a run
    /// whatever the thread count.
    ///
    /// Each slot owns its random stream, its quota of games and its row of the
    /// batch, so sharding decides only *who* steps a game, never what the game
    /// is. If this ever fails, the threading has introduced a shared mutable
    /// path and the measurements taken with it mean nothing -- which is why it
    /// asserts on the policy targets ply by ply and not merely on the results.
    #[test]
    fn driver_threads_do_not_change_the_games() {
        // concurrency 16 over 48 games, so every slot also exercises the refill
        // path: a quota, not a shared pool, is what keeps that reproducible.
        let one = run_flat_on(16, 48, 16, 1);
        for t in [2, 3, 4, 8] {
            same_games(&one, &run_flat_on(16, 48, 16, t), &format!("threads={t}"));
        }
    }

    /// Same, with playout cap randomization on, since PCR adds a second draw
    /// from the per-slot stream on every move.
    #[test]
    fn driver_threads_do_not_change_the_games_under_pcr() {
        let fast = Config { sims: 8, max_considered: 4, ..Config::default() };
        let pcr = Some(Pcr { prob: 0.25, fast });
        let one = run_pcr_on(pcr, 1);
        for t in [2, 4] {
            same_games(&one, &run_pcr_on(pcr, t), &format!("pcr threads={t}"));
        }
    }

    /// More threads than games is a configuration a benchmark sweep will hit.
    #[test]
    fn threads_are_clamped_to_the_slot_count() {
        let cfg = Config { sims: 16, max_considered: 7, ..Config::default() };
        let mut sp = SelfPlay::<Connect4>::new(4, 8, cfg, 42, 1234);
        assert_eq!(sp.set_threads(64), 4);
        assert_eq!(sp.set_threads(0), 1);
    }

    /// Every slot gets the same share of the games, give or take one.
    ///
    /// The old driver refilled from a shared counter, so a slot with short games
    /// could take several while another took none. That is invisible until it
    /// is parallel, and then it is a reproducibility bug.
    #[test]
    fn games_are_dealt_out_evenly_across_slots() {
        let cfg = Config { sims: 8, max_considered: 7, ..Config::default() };
        let sp = SelfPlay::<Connect4>::new(5, 23, cfg, 42, 1234);
        let shares: Vec<usize> = sp.slots.iter().map(|s| s.quota + 1).collect();
        assert_eq!(shares.iter().sum::<usize>(), 23);
        assert_eq!(shares, vec![5, 5, 5, 4, 4]);
    }

    #[test]
    fn plays_the_requested_number_of_games() {
        let games = run_flat(8, 20, 8);
        assert_eq!(games.len(), 20);
    }

    #[test]
    fn batches_are_the_size_of_the_live_game_count() {
        let cfg = Config {
            sims: 16,
            max_considered: 7,
            ..Config::default()
        };
        let mut sp = SelfPlay::<Connect4>::new(16, 64, cfg, 42, 7);
        let mut obs = Vec::new();

        let n = sp.next_batch(&mut obs);
        assert_eq!(n, 16, "every live game should contribute one leaf");
        assert_eq!(obs.len(), n * obs_len::<Connect4>());

        // Batches stay full while games remain to be started.
        let logits = vec![0.0f32; n * Connect4::POLICY_LEN];
        sp.submit(&logits, &vec![0.0f32; n]);
        assert_eq!(sp.next_batch(&mut obs), 16);
    }

    #[test]
    fn every_game_reaches_a_real_conclusion() {
        for t in run_flat(8, 24, 16) {
            assert!(t.plies > 0 && t.plies <= 42, "{} plies", t.plies);
            assert_eq!(t.samples.len(), t.plies);
        }
    }

    /// The sign convention that is easiest to get wrong and hardest to notice:
    /// a won game must give +1 to the winner's positions and -1 to the loser's,
    /// alternating ply by ply.
    #[test]
    fn outcomes_alternate_correctly_along_the_game() {
        for t in run_flat(8, 40, 16) {
            match t.winner {
                None => assert!(t.samples.iter().all(|s| s.z == 0.0), "draw with nonzero z"),
                Some(_) => {
                    for w in t.samples.windows(2) {
                        assert_eq!(w[0].z, -w[1].z, "z must alternate between plies");
                    }
                    // The player who made the last move is the winner, so the
                    // final sample must be a win from its own perspective.
                    assert_eq!(t.samples.last().unwrap().z, 1.0);
                }
            }
        }
    }

    #[test]
    fn policy_targets_are_normalised_and_legal() {
        for t in run_flat(4, 12, 16) {
            for s in &t.samples {
                let sum: f32 = s.policy.iter().map(|&(_, p)| p).sum();
                assert!((sum - 1.0).abs() < 1e-3, "policy sums to {sum}");
                let mut legal = Vec::new();
                s.pos.legal_moves(&mut legal);
                let legal_idx: Vec<u32> = legal
                    .iter()
                    .map(|&m| s.pos.policy_index(m) as u32)
                    .collect();
                for &(idx, _) in &s.policy {
                    assert!(
                        legal_idx.contains(&idx),
                        "policy mass on illegal move {idx}"
                    );
                }
            }
        }
    }

    #[test]
    fn results_are_reproducible_for_a_fixed_seed() {
        let a = run_flat(4, 8, 16);
        let b = run_flat(4, 8, 16);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.plies, y.plies);
            assert_eq!(x.winner, y.winner);
        }
    }

    #[test]
    fn random_play_produces_a_spread_of_results() {
        // With no network signal the games should not all end the same way; if
        // they do, something is stuck.
        let games = run_flat(16, 60, 8);
        let wins = games.iter().filter(|t| t.winner.is_some()).count();
        assert!(wins > 0, "no decisive games at all");
        let first_player_wins = games
            .iter()
            .filter(|t| t.winner == Some(Player::First))
            .count();
        assert!(first_player_wins > 0 && first_player_wins < games.len());
    }
}
