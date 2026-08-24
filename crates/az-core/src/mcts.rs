//! Gumbel AlphaZero search.
//!
//! Follows Danihelka et al., *Policy improvement by planning with Gumbel*
//! (ICLR 2022). Three things differ from the 2017 AlphaZero search, and all three
//! matter:
//!
//! 1. **Root actions are sampled without replacement** via the Gumbel-top-k
//!    trick, then narrowed by **sequential halving**, instead of being selected
//!    by PUCT. This is what gives the policy-improvement guarantee at small
//!    simulation counts.
//! 2. **Interior nodes** select by the completed-Q improved policy rather than
//!    PUCT.
//! 3. **The training target is the improved policy**, not the visit
//!    distribution. At 32 simulations a visit histogram is mostly noise; the
//!    improved policy is not.
//!
//! Together these let the search run at ~32 simulations instead of ~800, which
//! is roughly a 25x throughput multiplier and the difference between this project
//! being feasible on one machine and not.
//!
//! No Dirichlet noise and no temperature schedule: the Gumbel perturbation at the
//! root *is* the exploration mechanism.
//!
//! # Driving the search
//!
//! Search is resumable so that many games can pool their leaf evaluations into
//! one network call. Callers loop:
//!
//! ```text
//! while search.prepare() == Status::NeedsEval {
//!     search.pending().encode(&mut obs);   // ... batched with other games
//!     search.apply(&logits, value);
//! }
//! let (mv, target) = search.result();
//! ```

use crate::game::{Game, Outcome};

/// Not created yet.
const NO_CHILD: u32 = u32::MAX;

/// Ranking offset for a proven result.
///
/// Deliberately far outside the range any combination of Gumbel noise, policy
/// logit and sigma-transformed value can reach. A checkmate is not a very good
/// score; it is a different kind of thing, and the arithmetic should say so.
const PROVEN: f32 = 1e6;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Simulation budget per move (`n` in the paper).
    pub sims: u32,
    /// How many root actions to consider (`m`). Clamped to the legal count.
    pub max_considered: usize,
    /// Scale constants for the sigma transform.
    ///
    /// `c_scale` is 0.1, not 1.0, and it is applied to completed-Q values that
    /// have been rescaled to [0, 1]. Both details matter: with raw Q in [-1, 1]
    /// and a unit scale, sigma reaches ~60 while the logits are O(1), so the
    /// improved policy collapses onto argmax-Q and throws the prior away.
    pub c_visit: f32,
    pub c_scale: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sims: 32,
            max_considered: 16,
            c_visit: 50.0,
            c_scale: 0.1,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Status {
    /// [`Search::pending`] needs a network evaluation.
    NeedsEval,
    /// Budget spent; call [`Search::result`].
    Complete,
}

struct Edge<G: Game> {
    mv: G::Move,
    logit: f32,
    child: u32,
    visits: u32,
    /// Sum of backed-up values, from the *parent's* mover perspective.
    value_sum: f32,
    /// A result the rules guarantee, from the *parent's* mover perspective.
    ///
    /// `Some(Win)` means taking this move wins outright. This is not a statistic
    /// and must never be averaged with one -- see [`Search::solve`].
    proven: Option<Outcome>,
}

impl<G: Game> Edge<G> {
    #[inline]
    fn q(&self) -> Option<f32> {
        (self.visits > 0).then(|| self.value_sum / self.visits as f32)
    }
}

struct Node<G: Game> {
    pos: G,
    terminal: Option<Outcome>,
    expanded: bool,
    visits: u32,
    /// The network's value estimate here, from this node's mover perspective.
    value: f32,
    edges: Vec<Edge<G>>,
    /// A result the rules guarantee for this node's mover, once enough of the
    /// subtree below it has been solved.
    proven: Option<Outcome>,
}

pub struct Search<G: Game> {
    cfg: Config,
    nodes: Vec<Node<G>>,

    /// Gumbel perturbation per root edge, drawn once.
    gumbel: Vec<f32>,
    /// Root edges still in contention this phase.
    contenders: Vec<usize>,
    phases: u32,
    visits_each: u32,
    cursor: usize,
    visits_done: u32,
    sims_done: u32,

    /// Descent path awaiting backup: (node, edge taken).
    path: Vec<(u32, usize)>,
    pending: u32,
    finished: bool,

    scratch: Vec<G::Move>,
}

impl<G: Game> Search<G> {
    pub fn new(root: G, cfg: Config, rng: &mut Rng) -> Self {
        let terminal_root = {
            let mut buf = Vec::new();
            root.expand(&mut buf)
        };
        let mut s = Self {
            cfg,
            nodes: vec![Node {
                pos: root,
                terminal: terminal_root,
                expanded: false,
                visits: 0,
                value: 0.0,
                edges: Vec::new(),
                proven: terminal_root,
            }],
            gumbel: Vec::new(),
            contenders: Vec::new(),
            phases: 1,
            visits_each: 1,
            cursor: 0,
            visits_done: 0,
            sims_done: 0,
            path: Vec::new(),
            pending: 0,
            finished: false,
            scratch: Vec::new(),
        };
        s.seed_gumbel(rng);
        s
    }

    /// Gumbel noise is drawn up front so the ordering is fixed for the whole
    /// search -- resampling per phase would break the top-k sampling argument.
    fn seed_gumbel(&mut self, rng: &mut Rng) {
        // Sized once the root is expanded; store draws now so the search is
        // reproducible regardless of when expansion happens.
        self.gumbel = (0..G::POLICY_LEN).map(|_| rng.gumbel()).collect();
    }

    /// The position awaiting evaluation.
    pub fn pending(&self) -> &G {
        &self.nodes[self.pending as usize].pos
    }

    pub fn is_complete(&self) -> bool {
        self.finished
    }

    /// Advance until a network evaluation is required, or the search is done.
    pub fn prepare(&mut self) -> Status {
        if self.finished {
            return Status::Complete;
        }

        // The root always needs its own evaluation first: without logits there
        // is nothing to perturb and nothing to halve.
        if !self.nodes[0].expanded {
            if self.nodes[0].terminal.is_some() {
                self.finished = true;
                return Status::Complete;
            }
            self.pending = 0;
            self.path.clear();
            return Status::NeedsEval;
        }

        loop {
            let Some(root_edge) = self.next_root_edge() else {
                self.finished = true;
                return Status::Complete;
            };

            self.sims_done += 1;
            self.path.clear();

            // Descend: the root step is dictated by sequential halving, every
            // step below it by the completed-Q improved policy.
            let mut node = 0u32;
            let mut edge = root_edge;
            loop {
                self.path.push((node, edge));
                let child = self.nodes[node as usize].edges[edge].child;

                if child == NO_CHILD {
                    let created = self.create_child(node, edge);
                    if let Some(o) = self.nodes[created as usize].terminal {
                        // Terminal leaves carry their own exact value; no
                        // network call needed.
                        self.backup(created, o.value());
                        break;
                    }
                    self.pending = created;
                    return Status::NeedsEval;
                }

                if !self.nodes[child as usize].expanded {
                    // Reached before but never evaluated (can only happen for
                    // terminal nodes, which are handled at creation).
                    self.pending = child;
                    return Status::NeedsEval;
                }
                // A settled node has nothing left to teach. Back up the certainty
                // rather than spending a simulation inside it -- otherwise the
                // budget drains into lines whose result is already known.
                let settled = self.nodes[child as usize]
                    .terminal
                    .or(self.nodes[child as usize].proven);
                if let Some(o) = settled {
                    self.backup(child, o.value());
                    break;
                }

                node = child;
                edge = self.select_interior(node);
            }
        }
    }

    /// Feed back an evaluation for [`Search::pending`].
    ///
    /// `logits` is the full policy vector; only the legal entries are read.
    /// `value` is from the pending position's mover perspective, in [-1, 1].
    pub fn apply(&mut self, logits: &[f32], value: f32) {
        debug_assert_eq!(logits.len(), G::POLICY_LEN);
        let idx = self.pending;
        self.expand_node(idx, logits, value);

        if idx == 0 {
            self.start_root_phases();
            // The root evaluation is not a simulation; it does not count against
            // the budget and nothing is backed up through it.
            self.nodes[0].visits = 1;
            return;
        }
        self.backup(idx, value);
    }

    fn expand_node(&mut self, idx: u32, logits: &[f32], value: f32) {
        let node = &mut self.nodes[idx as usize];
        node.value = value;
        node.expanded = true;

        if node.terminal.is_some() {
            return;
        }

        node.pos.expand(&mut self.scratch);
        let pos = node.pos.clone();
        let edges = self
            .scratch
            .iter()
            .map(|&mv| Edge {
                mv,
                logit: logits[pos.policy_index(mv)],
                child: NO_CHILD,
                visits: 0,
                value_sum: 0.0,
                proven: None,
            })
            .collect();
        self.nodes[idx as usize].edges = edges;
    }

    fn create_child(&mut self, node: u32, edge: usize) -> u32 {
        let mut pos = self.nodes[node as usize].pos.clone();
        pos.play(self.nodes[node as usize].edges[edge].mv);
        let terminal = {
            let mut buf = Vec::new();
            pos.expand(&mut buf)
        };
        let id = self.nodes.len() as u32;
        self.nodes.push(Node {
            pos,
            terminal,
            expanded: terminal.is_some(),
            visits: 0,
            value: terminal.map_or(0.0, |o| o.value()),
            edges: Vec::new(),
            proven: terminal,
        });
        self.nodes[node as usize].edges[edge].child = id;
        // A terminal child settles this edge for good: the outcome is stated from
        // the child's mover's point of view, so it flips for the parent.
        if let Some(o) = terminal {
            self.nodes[node as usize].edges[edge].proven = Some(o.flip());
        }
        id
    }

    /// Propagate `value` (from `leaf`'s mover perspective) back to the root,
    /// flipping sign at each ply.
    fn backup(&mut self, leaf: u32, value: f32) {
        self.nodes[leaf as usize].visits += 1;
        let mut v = value;
        for i in (0..self.path.len()).rev() {
            let (node, edge) = self.path[i];
            v = -v; // now from `node`'s mover perspective
            let e = &mut self.nodes[node as usize].edges[edge];
            e.visits += 1;
            e.value_sum += v;
            self.nodes[node as usize].visits += 1;
        }
        self.propagate_proven();
    }

    /// Carry proven results up the path just walked.
    ///
    /// Runs after every backup and stops as soon as a node's status is unchanged,
    /// since nothing above it can change either.
    fn propagate_proven(&mut self) {
        for i in (0..self.path.len()).rev() {
            let (node, edge) = self.path[i];

            let child = self.nodes[node as usize].edges[edge].child;
            if child != NO_CHILD {
                if let Some(p) = self.nodes[child as usize].proven {
                    // The child states its result for its own mover; the parent
                    // sees the opposite.
                    self.nodes[node as usize].edges[edge].proven = Some(p.flip());
                }
            }

            let solved = self.solve(node);
            if solved == self.nodes[node as usize].proven {
                return;
            }
            self.nodes[node as usize].proven = solved;
        }
    }

    /// What the rules guarantee at `node`, given what its edges have proven.
    ///
    /// One winning move is enough to prove a win. Proving a loss needs *every*
    /// move settled and losing -- a single unexplored move might save the game.
    fn solve(&self, node: u32) -> Option<Outcome> {
        let n = &self.nodes[node as usize];
        if n.edges.is_empty() {
            return n.terminal;
        }
        let mut all_settled = true;
        let mut can_draw = false;
        for e in &n.edges {
            match e.proven {
                Some(Outcome::Win) => return Some(Outcome::Win),
                Some(Outcome::Draw) => can_draw = true,
                Some(Outcome::Loss) => {}
                None => all_settled = false,
            }
        }
        if !all_settled {
            return None;
        }
        Some(if can_draw {
            Outcome::Draw
        } else {
            Outcome::Loss
        })
    }

    // --- root: Gumbel top-k then sequential halving -------------------------

    fn start_root_phases(&mut self) {
        let n_legal = self.nodes[0].edges.len();
        let m = self.cfg.max_considered.min(n_legal).max(1);

        // Gumbel-top-k: sampling m actions without replacement from softmax(logits)
        // is exactly taking the top m of `logit + gumbel`.
        let mut ranked: Vec<usize> = (0..n_legal).collect();
        ranked.sort_by(|&a, &b| {
            self.root_gumbel_score(b)
                .partial_cmp(&self.root_gumbel_score(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked.truncate(m);
        self.contenders = ranked;

        self.phases = (usize::BITS - (m - 1).leading_zeros()).max(1);
        self.begin_phase();
    }

    /// `g(a) + logit(a)`, before any search information.
    #[inline]
    fn root_gumbel_score(&self, edge: usize) -> f32 {
        let e = &self.nodes[0].edges[edge];
        self.gumbel[self.nodes[0].pos.policy_index(e.mv)] + e.logit
    }

    /// `g(a) + logit(a) + sigma(q(a))` for every root edge, the ranking used for
    /// sequential halving and for the final choice.
    ///
    /// Computed for all edges at once because the sigma transform rescales
    /// across the node, so an edge's score is not a function of that edge alone.
    fn root_scores(&self) -> Vec<f32> {
        let sigma = self.sigma_completed(0);
        (0..self.nodes[0].edges.len())
            .map(|i| {
                let base = self.root_gumbel_score(i) + sigma[i];
                base + match self.nodes[0].edges[i].proven {
                    Some(Outcome::Win) => PROVEN,
                    Some(Outcome::Loss) => -PROVEN,
                    _ => 0.0,
                }
            })
            .collect()
    }

    fn begin_phase(&mut self) {
        let m = self.contenders.len() as u32;
        self.visits_each = (self.cfg.sims / (self.phases * m.max(1))).max(1);
        self.cursor = 0;
        self.visits_done = 0;
    }

    /// Which root edge the next simulation belongs to, or `None` when the
    /// budget is spent or a single action has won.
    fn next_root_edge(&mut self) -> Option<usize> {
        loop {
            if self.sims_done >= self.cfg.sims || self.contenders.len() <= 1 {
                return None;
            }
            if self.cursor >= self.contenders.len() {
                self.halve();
                continue;
            }
            let edge = self.contenders[self.cursor];
            self.visits_done += 1;
            if self.visits_done >= self.visits_each {
                self.cursor += 1;
                self.visits_done = 0;
            }
            return Some(edge);
        }
    }

    /// Drop the worse half of the contenders and start the next phase.
    fn halve(&mut self) {
        let scores = self.root_scores();
        let mut ranked = std::mem::take(&mut self.contenders);
        ranked.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let keep = ranked.len().div_ceil(2).max(1);
        ranked.truncate(keep);
        self.contenders = ranked;
        self.begin_phase();
    }

    // --- interior selection --------------------------------------------------

    /// `argmax_a [ pi'(a) - N(a) / (1 + sum_b N(b)) ]`.
    ///
    /// Picks whichever action the improved policy wants more of than it has
    /// already received -- a deterministic rule, with no exploration constant to
    /// tune.
    fn select_interior(&mut self, node: u32) -> usize {
        let improved = self.improved_policy(node);
        let n = &self.nodes[node as usize];
        let total: u32 = n.edges.iter().map(|e| e.visits).sum();
        let denom = 1.0 + total as f32;

        // Revisiting a settled move learns nothing, so spend the budget on moves
        // that are still open -- unless every move is settled.
        let all_settled = n.edges.iter().all(|e| e.proven.is_some());

        let mut best: Option<usize> = None;
        let mut best_score = f32::NEG_INFINITY;
        for (i, e) in n.edges.iter().enumerate() {
            if e.proven.is_some() && !all_settled {
                continue;
            }
            let score = improved[i] - e.visits as f32 / denom;
            if best.is_none() || score > best_score {
                best_score = score;
                best = Some(i);
            }
        }
        best.unwrap_or(0)
    }

    // --- the completed-Q machinery ------------------------------------------

    /// The sigma-transformed completed Q value for every edge of `node`.
    ///
    /// Three things happen here, and all three are load-bearing:
    ///
    /// 1. **Completion** -- edges the search never visited take `v_mix` rather
    ///    than the raw network value, so they are judged against what the search
    ///    has since learned.
    /// 2. **Rescaling** to [0, 1] across the node's edges. Without this the term
    ///    below dwarfs the logits and the improved policy degenerates to a
    ///    one-hot on argmax-Q.
    /// 3. **Visit scaling** -- `(c_visit + max_visits) * c_scale`, so the search
    ///    signal grows relative to the prior as evidence accumulates.
    fn sigma_completed(&self, node: u32) -> Vec<f32> {
        let priors = self.priors(node);
        let fallback = self.v_mix(node, &priors);
        let n = &self.nodes[node as usize];

        let mut q: Vec<f32> = n.edges.iter().map(|e| e.q().unwrap_or(fallback)).collect();
        rescale_unit(&mut q);

        let max_visits = n.edges.iter().map(|e| e.visits).max().unwrap_or(0);
        let scale = (self.cfg.c_visit + max_visits as f32) * self.cfg.c_scale;
        for v in &mut q {
            *v *= scale;
        }
        q
    }

    /// The value estimate used for actions the search never tried.
    ///
    /// Mixes the node's own network value with the visited children's Q,
    /// weighted by how much of the prior mass those children cover. Using the
    /// raw network value instead would systematically misrank unvisited actions
    /// once the search has learned something.
    fn v_mix(&self, node: u32, priors: &[f32]) -> f32 {
        let n = &self.nodes[node as usize];
        let total_visits: u32 = n.edges.iter().map(|e| e.visits).sum();
        if total_visits == 0 {
            return n.value;
        }

        let mut visited_prior = 0.0f32;
        let mut weighted_q = 0.0f32;
        for (i, e) in n.edges.iter().enumerate() {
            if let Some(q) = e.q() {
                visited_prior += priors[i];
                weighted_q += priors[i] * q;
            }
        }
        if visited_prior <= 0.0 {
            return n.value;
        }
        (n.value + total_visits as f32 * (weighted_q / visited_prior)) / (1.0 + total_visits as f32)
    }

    /// `pi' = softmax(logits + sigma(completedQ))`, with proven results honoured.
    ///
    /// When the search has proven a win, the training target says so outright.
    /// Softening it would teach the policy that a forced mate is merely a good
    /// idea.
    fn improved_policy(&self, node: u32) -> Vec<f32> {
        let n = &self.nodes[node as usize];

        let wins: Vec<usize> = n
            .edges
            .iter()
            .enumerate()
            .filter(|(_, e)| e.proven == Some(Outcome::Win))
            .map(|(i, _)| i)
            .collect();
        if !wins.is_empty() {
            let mut out = vec![0.0; n.edges.len()];
            let share = 1.0 / wins.len() as f32;
            for i in wins {
                out[i] = share;
            }
            return out;
        }

        let sigma = self.sigma_completed(node);
        let mut scored: Vec<f32> = n
            .edges
            .iter()
            .zip(&sigma)
            .map(|(e, &s)| e.logit + s)
            .collect();

        // Moves proven to lose get no mass -- unless every move loses, in which
        // case there is nothing left to prefer and the ordinary ranking stands.
        let losing: Vec<bool> = n
            .edges
            .iter()
            .map(|e| e.proven == Some(Outcome::Loss))
            .collect();
        if losing.iter().any(|&l| l) && !losing.iter().all(|&l| l) {
            for (i, &l) in losing.iter().enumerate() {
                if l {
                    scored[i] = f32::NEG_INFINITY;
                }
            }
        }
        softmax(&scored)
    }

    /// `softmax(logits)` over the legal moves at `node`.
    fn priors(&self, node: u32) -> Vec<f32> {
        let logits: Vec<f32> = self.nodes[node as usize]
            .edges
            .iter()
            .map(|e| e.logit)
            .collect();
        softmax(&logits)
    }

    // --- results -------------------------------------------------------------

    /// The chosen move and the improved-policy training target.
    ///
    /// The target is a full-length policy vector with zeros off the legal moves,
    /// ready to be a cross-entropy target against the network's masked output.
    pub fn result(&self) -> (G::Move, Vec<f32>) {
        let root = &self.nodes[0];
        assert!(root.expanded, "search never ran");

        let scores = self.root_scores();
        let best = *self
            .contenders
            .iter()
            .max_by(|&&a, &&b| {
                scores[a]
                    .partial_cmp(&scores[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("root has at least one legal move");

        let improved = self.improved_policy(0);
        let mut target = vec![0.0f32; G::POLICY_LEN];
        for (i, e) in root.edges.iter().enumerate() {
            target[root.pos.policy_index(e.mv)] = improved[i];
        }
        (root.edges[best].mv, target)
    }

    /// Root value estimate after search, from the root mover's perspective.
    pub fn root_value(&self) -> f32 {
        let priors = self.priors(0);
        self.v_mix(0, &priors)
    }

    pub fn simulations(&self) -> u32 {
        self.sims_done
    }

    pub fn nodes_allocated(&self) -> usize {
        self.nodes.len()
    }

    /// A result the rules guarantee at the root, if the search has found one.
    pub fn proven(&self) -> Option<Outcome> {
        self.nodes[0].proven
    }

    /// Per-root-move detail, ordered best first.
    ///
    /// Returns `(move, improved policy, visits, Q, proven)` for every legal root
    /// move. This is what a GUI needs for `info multipv`: it turns the search's
    /// opinion of *each* move into something a person can look at, rather than
    /// only the one it settled on.
    pub fn root_detail(&self) -> Vec<(G::Move, f32, u32, Option<f32>, Option<Outcome>)> {
        let improved = self.improved_policy(0);
        let scores = self.root_scores();
        let mut rows: Vec<(usize, f32)> = (0..self.nodes[0].edges.len())
            .map(|i| (i, scores[i]))
            .collect();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        rows.into_iter()
            .map(|(i, _)| {
                let e = &self.nodes[0].edges[i];
                (e.mv, improved[i], e.visits, e.q(), e.proven)
            })
            .collect()
    }
}

/// Min-max rescale to [0, 1] in place.
///
/// When every value is equal there is no preference to express, so the result is
/// flat zeros rather than an arbitrary spread.
fn rescale_unit(xs: &mut [f32]) {
    let min = xs.iter().copied().fold(f32::INFINITY, f32::min);
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    if range > 1e-8 {
        for x in xs.iter_mut() {
            *x = (*x - min) / range;
        }
    } else {
        xs.fill(0.0);
    }
}

fn softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![1.0 / xs.len() as f32; xs.len()];
    }
    let mut out: Vec<f32> = xs.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = out.iter().sum();
    if sum > 0.0 {
        for v in &mut out {
            *v /= sum;
        }
    }
    out
}

/// xorshift64*, seeded explicitly so self-play runs are reproducible.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in (0, 1).
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // 24 bits of mantissa, shifted off zero so the logs below stay finite.
        let bits = (self.next_u64() >> 40) as f32;
        (bits + 0.5) / (1u32 << 24) as f32
    }

    /// A standard Gumbel(0, 1) draw.
    #[inline]
    pub fn gumbel(&mut self) -> f32 {
        -(-self.next_f32().ln()).ln()
    }

    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connect4::Connect4;

    /// A stand-in network: uniform policy, and a value read straight off the
    /// board so tests exercise search rather than learning.
    fn flat_eval<G: Game>(_pos: &G) -> (Vec<f32>, f32) {
        (vec![0.0; G::POLICY_LEN], 0.0)
    }

    fn run<G: Game>(
        root: G,
        cfg: Config,
        seed: u64,
        eval: impl Fn(&G) -> (Vec<f32>, f32),
    ) -> Search<G> {
        let mut rng = Rng::new(seed);
        let mut s = Search::new(root, cfg, &mut rng);
        while s.prepare() == Status::NeedsEval {
            let (logits, value) = eval(s.pending());
            s.apply(&logits, value);
        }
        s
    }

    #[test]
    fn respects_the_simulation_budget() {
        for sims in [2u32, 8, 32, 64, 200] {
            let cfg = Config {
                sims,
                ..Config::default()
            };
            let s = run(Connect4::new(), cfg, 7, flat_eval);
            assert!(
                s.simulations() <= sims,
                "used {} of {sims}",
                s.simulations()
            );
        }
    }

    #[test]
    fn produces_a_normalised_policy_target() {
        let s = run(Connect4::new(), Config::default(), 11, flat_eval);
        let (mv, target) = s.result();
        assert!(mv < 7);
        assert_eq!(target.len(), Connect4::POLICY_LEN);
        let sum: f32 = target.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "target sums to {sum}");
        assert!(target.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn target_is_zero_on_illegal_moves() {
        // Fill column 0 so it is not playable.
        let mut pos = Connect4::new();
        for _ in 0..6 {
            pos.play(0);
        }
        let s = run(pos, Config::default(), 3, flat_eval);
        let (_, target) = s.result();
        assert_eq!(target[0], 0.0, "illegal column carries probability");
        let sum: f32 = target.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    /// The sharpest correctness check available without a trained network: with
    /// a *uniform* policy and no value signal, search alone must still find a
    /// move that wins on the spot.
    #[test]
    fn finds_an_immediate_win() {
        // First player has three in column 3 with the fourth available.
        let pos = Connect4::from_moves(&[3, 0, 3, 1, 3, 2]).unwrap();
        assert!(pos.is_winning_move(3), "test position is wrong");

        let cfg = Config {
            sims: 64,
            max_considered: 7,
            ..Config::default()
        };
        for seed in 0..20 {
            let s = run(pos, cfg, seed, flat_eval);
            let (mv, _) = s.result();
            assert_eq!(mv, 3, "seed {seed} missed the winning move");
        }
    }

    /// Symmetrically: it must see the opponent's threat and block it.
    #[test]
    fn blocks_an_immediate_threat() {
        // Second player to move; first player threatens to complete column 3.
        let pos = Connect4::from_moves(&[3, 0, 3, 1, 3]).unwrap();
        let cfg = Config {
            sims: 128,
            max_considered: 7,
            ..Config::default()
        };
        let mut blocked = 0;
        for seed in 0..20 {
            let s = run(pos, cfg, seed, flat_eval);
            let (mv, _) = s.result();
            if mv == 3 {
                blocked += 1;
            }
        }
        assert!(blocked >= 18, "only blocked {blocked}/20 times");
    }

    #[test]
    fn a_forced_position_returns_its_only_move() {
        // Every column full except one.
        let mut pos = Connect4::new();
        for col in 0..6 {
            for _ in 0..6 {
                if pos.outcome().is_some() {
                    break;
                }
                pos.play(col as u8);
            }
        }
        if pos.outcome().is_none() {
            let mut legal = Vec::new();
            pos.legal_moves(&mut legal);
            if legal.len() == 1 {
                let s = run(pos, Config::default(), 5, flat_eval);
                let (mv, _) = s.result();
                assert_eq!(mv, legal[0]);
            }
        }
    }

    /// Search must exploit a value signal even when the policy is uniform.
    #[test]
    fn follows_the_value_signal() {
        // Reward the mover for holding the centre column.
        fn centre_eval(pos: &Connect4) -> (Vec<f32>, f32) {
            let mut v: f32 = 0.0;
            for row in 0..6 {
                match pos.cell(row, 3) {
                    'x' => v += 0.2,
                    'o' => v -= 0.2,
                    _ => {}
                }
            }
            (vec![0.0; 7], v.clamp(-1.0, 1.0))
        }

        let cfg = Config {
            sims: 64,
            max_considered: 7,
            ..Config::default()
        };
        let mut centre = 0;
        for seed in 0..20 {
            let s = run(Connect4::new(), cfg, seed, centre_eval);
            if s.result().0 == 3 {
                centre += 1;
            }
        }
        assert!(centre >= 15, "took the centre only {centre}/20 times");
    }

    #[test]
    fn gumbel_draws_look_like_gumbel() {
        let mut rng = Rng::new(42);
        let n = 200_000;
        let mut sum = 0.0f64;
        for _ in 0..n {
            let g = rng.gumbel();
            assert!(g.is_finite(), "gumbel produced {g}");
            sum += g as f64;
        }
        // Mean of Gumbel(0,1) is Euler-Mascheroni, ~0.5772.
        let mean = sum / n as f64;
        assert!((mean - 0.5772).abs() < 0.02, "mean was {mean}");
    }

    #[test]
    fn rescale_maps_to_the_unit_interval() {
        let mut v = vec![-1.0, 0.0, 1.0];
        rescale_unit(&mut v);
        assert_eq!(v, vec![0.0, 0.5, 1.0]);

        // All equal means no preference to express, not an arbitrary spread.
        let mut flat = vec![0.7, 0.7, 0.7];
        rescale_unit(&mut flat);
        assert_eq!(flat, vec![0.0, 0.0, 0.0]);

        let mut one = vec![3.0];
        rescale_unit(&mut one);
        assert_eq!(one, vec![0.0]);
    }

    /// When search has learned nothing, the improved policy must be the prior.
    ///
    /// This is the invariant that the sigma transform broke: with raw Q values
    /// and a unit scale, the term reached ~60 against O(1) logits and the target
    /// collapsed onto argmax-Q, discarding the network's policy entirely.
    #[test]
    fn an_uninformative_search_returns_the_prior() {
        // Strongly peaked prior on column 5, and a value function that says
        // nothing at all.
        fn peaked(_pos: &Connect4) -> (Vec<f32>, f32) {
            let mut logits = vec![0.0; 7];
            logits[5] = 3.0;
            (logits, 0.0)
        }

        let cfg = Config {
            sims: 32,
            max_considered: 7,
            ..Config::default()
        };
        let s = run(Connect4::new(), cfg, 4, peaked);
        let (_, target) = s.result();

        // softmax([0,0,0,0,0,3,0]) puts ~0.71 on column 5.
        let expected = 3.0f32.exp() / (3.0f32.exp() + 6.0);
        assert!(
            (target[5] - expected).abs() < 0.05,
            "prior was distorted: {:.3} vs expected {:.3}",
            target[5],
            expected
        );
    }

    /// With a uniform prior and a wide spread of Q values, the target should be
    /// peaked but not degenerate -- the sigma term is bounded by construction.
    #[test]
    fn the_target_stays_a_distribution_not_a_one_hot() {
        // Values that differ sharply by position, so Q spans a wide range.
        fn spread(pos: &Connect4) -> (Vec<f32>, f32) {
            let v = if pos.plies().is_multiple_of(2) {
                0.9
            } else {
                -0.9
            };
            (vec![0.0; 7], v)
        }

        let cfg = Config {
            sims: 32,
            max_considered: 7,
            ..Config::default()
        };
        let s = run(Connect4::new(), cfg, 9, spread);
        let (_, target) = s.result();
        let max = target.iter().copied().fold(0.0f32, f32::max);
        assert!(
            max < 0.999,
            "target collapsed to a one-hot ({max:.4}); sigma is unbounded again"
        );
        let sum: f32 = target.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    /// The test the whole change exists for: a certain win must beat a confident
    /// wrong opinion. The network here is adversarial -- it puts a large logit on
    /// a losing move and none on the winning one.
    #[test]
    fn a_proven_win_beats_a_confidently_wrong_network() {
        let pos = Connect4::from_moves(&[3, 0, 3, 1, 3, 2]).unwrap();
        assert!(pos.is_winning_move(3), "test position is wrong");

        fn adversarial(p: &Connect4) -> (Vec<f32>, f32) {
            let mut logits = vec![0.0; 7];
            logits[3] = -8.0; // hates the winning move
            logits[6] = 8.0; // loves a pointless one
                             // ...and is confidently wrong about the value, too
            (
                logits,
                if p.plies().is_multiple_of(2) {
                    0.95
                } else {
                    -0.95
                },
            )
        }

        let cfg = Config {
            sims: 32,
            max_considered: 7,
            ..Config::default()
        };
        for seed in 0..25 {
            let s = run(pos, cfg, seed, adversarial);
            assert_eq!(
                s.result().0,
                3,
                "seed {seed} was talked out of a forced win"
            );
            assert_eq!(s.proven(), Some(Outcome::Win));
        }
    }

    /// The training target should state the proven result, not hedge it.
    #[test]
    fn a_proven_win_becomes_the_policy_target() {
        let pos = Connect4::from_moves(&[3, 0, 3, 1, 3, 2]).unwrap();
        let cfg = Config {
            sims: 32,
            max_considered: 7,
            ..Config::default()
        };
        let s = run(pos, cfg, 3, flat_eval);
        let (_, target) = s.result();
        assert!(
            target[3] > 0.99,
            "target hedged a forced win: {:.3}",
            target[3]
        );
    }

    /// The mirror: a move that hands the opponent an immediate win must be
    /// avoided.
    ///
    /// Proving a *loss* is strictly harder than proving a win. A winning move is
    /// itself terminal, so one visit settles it. A losing move is not: the search
    /// has to descend into it, expand it, and reach the opponent's winning reply
    /// before the edge is settled -- and it must do that for *every* alternative
    /// before the position is solved.
    ///
    /// So it is budget-dependent, and sharply so. Measured over 100 seeds on this
    /// position: 16 sims blocks 18% of the time, 32 sims 40%, 64 sims 62%, and
    /// 128 sims 100% with the target fully settled. The threshold below is that
    /// measurement, not a guess.
    #[test]
    fn a_proven_loss_is_avoided() {
        // Second player to move; first player threatens to complete column 3.
        let pos = Connect4::from_moves(&[3, 0, 3, 1, 3]).unwrap();
        let cfg = Config {
            sims: 128,
            max_considered: 7,
            ..Config::default()
        };
        for seed in 0..40 {
            let s = run(pos, cfg, seed, flat_eval);
            let (mv, target) = s.result();
            assert_eq!(mv, 3, "seed {seed} allowed a forced loss");
            assert!(
                target[3] > 0.9,
                "seed {seed}: target hedged a solved position ({:.3})",
                target[3]
            );
        }
    }

    /// A position where every move loses must still return a legal move rather
    /// than falling over.
    #[test]
    fn a_lost_position_still_returns_a_move() {
        // Column 3 has three in a row for the opponent and two ways to finish.
        let pos = Connect4::from_moves(&[3, 0, 3, 1, 3, 6]).unwrap();
        let cfg = Config {
            sims: 32,
            max_considered: 7,
            ..Config::default()
        };
        let s = run(pos, cfg, 1, flat_eval);
        let (mv, target) = s.result();
        let mut legal = Vec::new();
        pos.legal_moves(&mut legal);
        assert!(legal.contains(&mv));
        let sum: f32 = target.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "target sums to {sum}");
    }

    #[test]
    fn terminal_root_completes_without_evaluation() {
        let pos = Connect4::from_moves(&[3, 4, 3, 4, 3, 4, 3]).unwrap();
        assert!(pos.outcome().is_some());
        let mut rng = Rng::new(1);
        let mut s = Search::new(pos, Config::default(), &mut rng);
        assert_eq!(s.prepare(), Status::Complete);
    }
}
