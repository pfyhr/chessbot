//! The game abstraction that MCTS and self-play are written against.
//!
//! Deliberately narrow. Everything MCTS needs and nothing it does not, so that
//! adding a game is a small, obviously-correct amount of work.

/// Which side is to move. Two-player zero-sum games only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Player {
    First,
    Second,
}

impl Player {
    #[inline]
    pub fn other(self) -> Self {
        match self {
            Player::First => Player::Second,
            Player::Second => Player::First,
        }
    }
}

/// Terminal result, always from the perspective of the player to move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Win,
    Loss,
    Draw,
}

impl Outcome {
    /// Scalar value in [-1, 1] for the side to move.
    #[inline]
    pub fn value(self) -> f32 {
        match self {
            Outcome::Win => 1.0,
            Outcome::Loss => -1.0,
            Outcome::Draw => 0.0,
        }
    }

    /// Flip to the opponent's perspective.
    #[inline]
    pub fn flip(self) -> Self {
        match self {
            Outcome::Win => Outcome::Loss,
            Outcome::Loss => Outcome::Win,
            Outcome::Draw => Outcome::Draw,
        }
    }
}

/// A two-player, zero-sum, perfect-information game.
///
/// Implementors must be cheap to clone: MCTS clones positions constantly, and in
/// chess this is the single hottest allocation in the engine.
pub trait Game: Clone + Send + 'static {
    /// A move. Small and `Copy` -- these live in move lists on the stack.
    type Move: Copy + Eq + std::fmt::Debug + Send;

    /// Size of the flat policy vector. Every move maps to a distinct index below
    /// this bound. For chess this is 4672 (73 move-type planes x 64 squares).
    const POLICY_LEN: usize;

    /// Shape of the observation tensor handed to the network, as (channels, height, width).
    const OBS_SHAPE: (usize, usize, usize);

    fn initial() -> Self;

    fn player_to_move(&self) -> Player;

    /// Fill `out` with the legal moves *and* report whether the game has ended.
    ///
    /// These are deliberately one call rather than two. MCTS needs both at every
    /// node it visits, and asking separately makes the position generate its
    /// moves more than once: shakmaty's `is_checkmate` and `is_stalemate` each
    /// run a full move generation internally, so the two-call spelling costs
    /// three movegens per node where one suffices. Measured at 1.80x on chess --
    /// see `src/bin/nodecost.rs`.
    ///
    /// **A `Some` result means the node is terminal even if `out` is non-empty.**
    /// A draw by the fifty-move rule or by repetition has legal moves available
    /// and is still over. Callers must check the outcome before the move list.
    fn expand(&self, out: &mut Vec<Self::Move>) -> Option<Outcome>;

    /// Apply a move that is known to be legal.
    fn play(&mut self, mv: Self::Move);

    /// Legal moves only. Convenience for tests and tooling.
    ///
    /// Search code should call [`Game::expand`] instead and use both halves of
    /// the answer.
    fn legal_moves(&self, out: &mut Vec<Self::Move>) {
        self.expand(out);
    }

    /// `Some(outcome)` if the game has ended, from the side-to-move's perspective.
    ///
    /// Allocates, and throws away the move list it had to build. Convenience for
    /// tests and tooling; search code should call [`Game::expand`].
    fn outcome(&self) -> Option<Outcome> {
        let mut scratch = Vec::new();
        self.expand(&mut scratch)
    }

    /// Index of `mv` in the flat policy vector. Must be injective over the legal
    /// moves of any single position, and must round-trip with [`Game::move_from_policy_index`].
    fn policy_index(&self, mv: Self::Move) -> usize;

    /// Inverse of [`Game::policy_index`] for a move legal in this position.
    fn move_from_policy_index(&self, index: usize) -> Option<Self::Move>;

    /// Write the observation planes into `out`, which is exactly `OBS_SHAPE`
    /// elements long. Always from the side-to-move's perspective.
    fn encode(&self, out: &mut [f32]);
}

/// Total number of floats in one observation.
pub const fn obs_len<G: Game>() -> usize {
    let (c, h, w) = G::OBS_SHAPE;
    c * h * w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn player_alternates() {
        assert_eq!(Player::First.other(), Player::Second);
        assert_eq!(Player::First.other().other(), Player::First);
    }

    #[test]
    fn outcome_flip_is_an_involution() {
        for o in [Outcome::Win, Outcome::Loss, Outcome::Draw] {
            assert_eq!(o.flip().flip(), o);
            assert_eq!(o.flip().value(), -o.value());
        }
    }
}
