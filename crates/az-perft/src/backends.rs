//! Perft implementations over each candidate rules library.
//!
//! Both backends implement the identical algorithm behind the same trait so the
//! timing comparison is fair. Static dispatch via an associated const means the
//! abstraction costs nothing at runtime.

/// A chess rules library we are evaluating as the engine's movegen core.
pub trait PerftBackend {
    const NAME: &'static str;
    /// Crate version, for the benchmark record.
    const VERSION: &'static str;

    /// Count leaf nodes at `depth` from `fen`.
    ///
    /// `bulk` enables bulk counting: at depth 1, return the move count directly
    /// instead of making and unmaking each move. This is the standard perft
    /// optimisation and typically buys 3-5x, but it exercises less of the
    /// make-move path -- which is the path self-play actually cares about. We
    /// measure both.
    fn perft(fen: &str, depth: u32, bulk: bool) -> u64;
}

// ---------------------------------------------------------------------------
// shakmaty
// ---------------------------------------------------------------------------

pub struct Shakmaty;

mod shak_impl {
    use shakmaty::fen::Fen;
    use shakmaty::{CastlingMode, Chess, Position};

    pub fn parse(fen: &str) -> Chess {
        let parsed: Fen = fen.parse().expect("valid FEN");
        parsed
            .into_position(CastlingMode::Standard)
            .expect("legal position")
    }

    pub fn go(pos: &Chess, depth: u32, bulk: bool) -> u64 {
        if depth == 0 {
            return 1;
        }
        let moves = pos.legal_moves();
        if bulk && depth == 1 {
            return moves.len() as u64;
        }
        let mut nodes = 0;
        for m in &moves {
            let mut child = pos.clone();
            child.play_unchecked(*m);
            nodes += go(&child, depth - 1, bulk);
        }
        nodes
    }
}

impl PerftBackend for Shakmaty {
    const NAME: &'static str = "shakmaty";
    const VERSION: &'static str = "0.30.1";

    fn perft(fen: &str, depth: u32, bulk: bool) -> u64 {
        let pos = shak_impl::parse(fen);
        shak_impl::go(&pos, depth, bulk)
    }
}

// ---------------------------------------------------------------------------
// cozy-chess
// ---------------------------------------------------------------------------

pub struct CozyChess;

mod cozy_impl {
    use cozy_chess::Board;

    pub fn parse(fen: &str) -> Board {
        Board::from_fen(fen, false).expect("valid FEN")
    }

    pub fn go(board: &Board, depth: u32, bulk: bool) -> u64 {
        if depth == 0 {
            return 1;
        }
        let mut nodes = 0u64;
        // cozy-chess yields moves grouped per piece rather than one at a time;
        // that grouping is where a lot of its speed comes from, so we consume it
        // in its native shape rather than flattening to a Vec.
        board.generate_moves(|piece_moves| {
            if bulk && depth == 1 {
                nodes += piece_moves.len() as u64;
            } else {
                for mv in piece_moves {
                    let mut child = board.clone();
                    child.play_unchecked(mv);
                    nodes += go(&child, depth - 1, bulk);
                }
            }
            false // false = keep generating
        });
        nodes
    }
}

impl PerftBackend for CozyChess {
    const NAME: &'static str = "cozy-chess";
    const VERSION: &'static str = "0.3.4";

    fn perft(fen: &str, depth: u32, bulk: bool) -> u64 {
        let board = cozy_impl::parse(fen);
        cozy_impl::go(&board, depth, bulk)
    }
}
