//! Standard perft test positions with known-correct node counts.
//!
//! These are the canonical positions from the Chess Programming Wiki. They are chosen
//! to exercise the rules that movegen implementations get wrong: en passant pins,
//! castling through check, underpromotion, and discovered check.
//!
//! Source: https://www.chessprogramming.org/Perft_Results

pub struct TestPosition {
    pub name: &'static str,
    pub fen: &'static str,
    /// Expected node counts, indexed so that `counts[i]` is perft(i + 1).
    pub counts: &'static [u64],
}

pub const POSITIONS: &[TestPosition] = &[
    TestPosition {
        name: "startpos",
        fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        counts: &[
            20,
            400,
            8_902,
            197_281,
            4_865_609,
            119_060_324,
            3_195_901_860,
        ],
    },
    TestPosition {
        // The classic stress test: castling, en passant, promotions, pins all at once.
        name: "kiwipete",
        fen: "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        counts: &[48, 2_039, 97_862, 4_085_603, 193_690_690, 8_031_647_685],
    },
    TestPosition {
        // Sparse endgame; catches en-passant-discovered-check bugs.
        name: "position-3",
        fen: "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        counts: &[14, 191, 2_812, 43_238, 674_624, 11_030_083, 178_633_661],
    },
    TestPosition {
        name: "position-4",
        fen: "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        counts: &[6, 264, 9_467, 422_333, 15_833_292, 706_045_033],
    },
    TestPosition {
        name: "position-5",
        fen: "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        counts: &[44, 1_486, 62_379, 2_103_487, 89_941_194],
    },
    TestPosition {
        name: "position-6",
        fen: "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        counts: &[46, 2_079, 89_890, 3_894_594, 164_075_551, 6_923_051_137],
    },
];

impl TestPosition {
    /// Expected node count for `depth`, if we have a reference value for it.
    pub fn expected(&self, depth: usize) -> Option<u64> {
        if depth == 0 {
            return Some(1);
        }
        self.counts.get(depth - 1).copied()
    }
}

pub fn by_name(name: &str) -> Option<&'static TestPosition> {
    POSITIONS.iter().find(|p| p.name == name)
}
