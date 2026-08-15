//! Core of the AlphaZero-style engine: the game abstraction, position encodings,
//! Gumbel MCTS, and the self-play driver.
//!
//! The crate is generic over [`Game`] so the whole reinforcement-learning loop can
//! be validated on Connect4 -- which converges to near-perfect play in an
//! afternoon -- before being pointed at chess, where a subtly wrong training
//! target is indistinguishable from slow learning for weeks.

pub mod chess;
pub mod game;

pub use chess::ChessPos;
pub use game::{Game, Outcome, Player};
