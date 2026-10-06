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

use crate::game::{obs_len, Game, Outcome, Player};
use crate::mcts::{Config, Rng, Search, Status};

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

struct Slot<G: Game> {
    pos: G,
    search: Search<G>,
    samples: Vec<Sample<G>>,
    live: bool,
    /// Whether the search currently in flight was given the full budget.
    full: bool,
    /// This slot's own random stream, derived from the master seed and the slot
    /// index.
    ///
    /// A single shared generator would make the games depend on the *order* in
    /// which slots consume it, which is fine while the driver is sequential and
    /// fatal once it is not: thread scheduling would decide the Gumbel noise and
    /// a seed would no longer reproduce a run. Per-slot streams make slot order
    /// irrelevant, which is what lets the loops below be parallel at all.
    rng: Rng,
}

pub struct SelfPlay<G: Game> {
    cfg: Config,
    /// `None` reproduces the pre-PCR behaviour exactly: every move full budget,
    /// every position a policy target.
    pcr: Option<Pcr>,
    max_plies: usize,
    slots: Vec<Slot<G>>,

    /// Games still to be started beyond those already in flight.
    remaining: usize,
    /// Slot index for each row of the batch produced by `next_batch`.
    batch_slots: Vec<usize>,

    finished: Vec<Trajectory<G>>,
    games_completed: usize,
    evaluations: u64,

    /// Running tallies, kept separate from `finished` so that reading stats
    /// never consumes the training data (and vice versa).
    first_wins: usize,
    second_wins: usize,
    draws: usize,
    total_plies: usize,
    truncated: usize,
}

impl<G: Game> SelfPlay<G> {
    /// `concurrency` games in flight at once; `total_games` played in all.
    ///
    /// Finished games are replaced immediately so the batch stays full: letting
    /// it drain would spend the tail of every run at a batch size where the GPU
    /// is idle.
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
                let full = Self::roll(&mut srng, pcr);
                Slot {
                    pos: G::initial(),
                    search: Search::new(G::initial(), Self::budget(cfg, pcr, full), &mut srng),
                    samples: Vec::new(),
                    live: true,
                    full,
                    rng: srng,
                }
            })
            .collect();

        Self {
            cfg,
            pcr,
            max_plies,
            slots,
            remaining: total_games.saturating_sub(concurrency),
            batch_slots: Vec::new(),
            finished: Vec::new(),
            games_completed: 0,
            evaluations: 0,
            first_wins: 0,
            second_wins: 0,
            draws: 0,
            total_plies: 0,
            truncated: 0,
        }
    }

    /// Step every live game to its next leaf and encode them all.
    ///
    /// Returns the batch size; zero means every game has finished. `out` is
    /// filled with `n * obs_len` floats in row-major `(N, C, H, W)` order.
    pub fn next_batch(&mut self, out: &mut Vec<f32>) -> usize {
        let obs = obs_len::<G>();
        out.clear();
        self.batch_slots.clear();

        // Sequential on purpose. The loops below are over independent games and
        // look obviously parallel, and on 2026-10-06 they were parallelised with
        // rayon to attack the driver's 29s of a 92s generation on an RTX 3090.
        // Measured, it was *slower*: next_batch went 4.8s -> 5.4s at 4 threads
        // and 5.3s at 32. A call only carries ~5ms of work across at most 256
        // slots, so waking a pool costs more than the work it hands out.
        //
        // The per-slot RNG introduced for that attempt is kept: it removes a
        // real order dependence, and it is the prerequisite if this is ever
        // retried at a granularity where threads pay.
        for i in 0..self.slots.len() {
            if !self.slots[i].live {
                continue;
            }
            loop {
                match self.slots[i].search.prepare() {
                    Status::NeedsEval => {
                        let start = out.len();
                        out.resize(start + obs, 0.0);
                        self.slots[i].search.pending().encode(&mut out[start..]);
                        self.batch_slots.push(i);
                        break;
                    }
                    Status::Complete => {
                        // Loop rather than break even when the game ends: the
                        // slot has just been refilled with a fresh game, and
                        // that game's root evaluation belongs in *this* batch.
                        self.commit_move(i);
                        if !self.slots[i].live {
                            break;
                        }
                    }
                }
            }
        }

        self.evaluations += self.batch_slots.len() as u64;
        self.batch_slots.len()
    }

    /// Feed back one forward pass.
    ///
    /// `logits` is `n * POLICY_LEN`, `values` is `n`, both in the order
    /// [`SelfPlay::next_batch`] produced.
    pub fn submit(&mut self, logits: &[f32], values: &[f32]) {
        let n = self.batch_slots.len();
        assert_eq!(values.len(), n, "expected {n} values");
        assert_eq!(logits.len(), n * G::POLICY_LEN, "expected {n} policy rows");

        for (k, &slot) in self.batch_slots.iter().enumerate() {
            let row = &logits[k * G::POLICY_LEN..(k + 1) * G::POLICY_LEN];
            self.slots[slot].search.apply(row, values[k]);
        }
    }

    /// Play the move the search settled on.
    ///
    /// If that ends the game, the slot is either refilled with a fresh game or
    /// marked dead; callers check `slots[i].live` rather than a return value.
    fn commit_move(&mut self, i: usize) {
        let (mv, target) = self.slots[i].search.result();
        let root_value = self.slots[i].search.root_value();

        let policy: Vec<(u32, f32)> = target
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(idx, &p)| (idx as u32, p))
            .collect();

        let pos = self.slots[i].pos.clone();
        let policy_mask = if self.slots[i].full { 1.0 } else { 0.0 };
        self.slots[i].samples.push(Sample {
            pos,
            policy,
            z: 0.0,
            root_value,
            policy_mask,
        });

        self.slots[i].pos.play(mv);

        let mut scratch = Vec::new();
        let outcome = self.slots[i].pos.expand(&mut scratch);
        let over = outcome.is_some() || self.slots[i].samples.len() >= self.max_plies;

        if over {
            self.finish_game(i, outcome);
            return;
        }

        let next = self.slots[i].pos.clone();
        let pcr = self.pcr;
        let base = self.cfg;
        let slot = &mut self.slots[i];
        let full = Self::roll(&mut slot.rng, pcr);
        slot.full = full;
        slot.search = Search::new(next, Self::budget(base, pcr, full), &mut slot.rng);
    }

    fn finish_game(&mut self, i: usize, outcome: Option<Outcome>) {
        // `None` here means the ply limit stopped the game, not that it was drawn.
        let decided = outcome.is_some();
        let final_mover = self.slots[i].pos.player_to_move();
        let winner = match outcome {
            Some(Outcome::Loss) => Some(final_mover.other()),
            Some(Outcome::Win) => Some(final_mover),
            // A draw, or a game cut short by the ply limit.
            _ => None,
        };

        let mut samples = std::mem::take(&mut self.slots[i].samples);
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

        // Refill the slot so the next batch stays full.
        if self.remaining > 0 {
            self.remaining -= 1;
            self.slots[i].pos = G::initial();
            let pcr = self.pcr;
            let base = self.cfg;
            let slot = &mut self.slots[i];
            let full = Self::roll(&mut slot.rng, pcr);
            slot.full = full;
            slot.search = Search::new(G::initial(), Self::budget(base, pcr, full), &mut slot.rng);
            self.slots[i].samples.clear();
        } else {
            self.slots[i].live = false;
        }
    }

    pub fn is_done(&self) -> bool {
        self.slots.iter().all(|s| !s.live)
    }

    pub fn games_completed(&self) -> usize {
        self.games_completed
    }

    /// Total leaf evaluations requested so far.
    pub fn evaluations(&self) -> u64 {
        self.evaluations
    }

    /// `(first_wins, second_wins, draws, mean_plies)` over every game so far.
    ///
    /// Reading this does not consume anything.
    pub fn stats(&self) -> (usize, usize, usize, f64) {
        let n = self.games_completed.max(1);
        (
            self.first_wins,
            self.second_wins,
            self.draws,
            self.total_plies as f64 / n as f64,
        )
    }

    /// How many finished games ran out of plies instead of reaching a result.
    pub fn truncated(&self) -> usize {
        self.truncated
    }

    pub fn take_finished(&mut self) -> Vec<Trajectory<G>> {
        std::mem::take(&mut self.finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connect4::Connect4;

    /// As `run_flat`, with playout cap randomization.
    fn run_pcr(pcr: Option<Pcr>) -> Vec<Trajectory<Connect4>> {
        let cfg = Config { sims: 64, max_considered: 7, ..Config::default() };
        let mut sp = SelfPlay::<Connect4>::with_pcr(8, 24, cfg, 42, 1234, pcr);
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
        let cfg = Config {
            sims,
            max_considered: 7,
            ..Config::default()
        };
        let mut sp = SelfPlay::<Connect4>::new(concurrency, total, cfg, 42, 1234);
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
