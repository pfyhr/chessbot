//! Core of the AlphaZero-style engine: the game abstraction, position encodings,
//! Gumbel MCTS, and the self-play driver.
//!
//! The crate is generic over [`Game`] so the whole reinforcement-learning loop can
//! be validated on Connect4 -- which converges to near-perfect play in an
//! afternoon -- before being pointed at chess, where a subtly wrong training
//! target is indistinguishable from slow learning for weeks.

pub mod chess;
pub mod connect4;
pub mod game;
pub mod mcts;
pub mod prof;
pub mod selfplay;

pub use chess::ChessPos;
pub use connect4::Connect4;
pub use game::{Game, Outcome, Player};
