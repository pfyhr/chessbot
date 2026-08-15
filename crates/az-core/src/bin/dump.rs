//! Dumps positions and their move encodings as TSV, for cross-checking against
//! an independent implementation in Python.
//!
//! Self-consistency tests in `chess.rs` prove the encoding is a bijection. They
//! cannot prove it is *AlphaZero's* bijection -- a coherently wrong plane layout
//! passes them all. This binary exists so a separately-written Python encoder can
//! disagree with us.
//!
//! Output, one line per position:
//!
//! ```text
//! <fen>\t<uci>:<index>,<uci>:<index>,...
//! ```
//!
//! UCI move strings are emitted with castling as the king's true two-square
//! destination (e1g1), matching python-chess's default, rather than shakmaty's
//! king-takes-rook convention.

use az_core::chess::ChessPos;
use az_core::Game;
use shakmaty::Move;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let games: usize = arg(&args, "--games").unwrap_or(300);
    let max_plies: usize = arg(&args, "--max-plies").unwrap_or(140);
    let seed: u64 = arg(&args, "--seed").unwrap_or(0x9E3779B97F4A7C15);

    let mut rng = Rng::new(seed);
    let mut legal = Vec::new();

    for _ in 0..games {
        let mut pos = ChessPos::new();
        for _ in 0..max_plies {
            if pos.outcome().is_some() {
                break;
            }
            pos.legal_moves(&mut legal);
            if legal.is_empty() {
                break;
            }

            let encoded: Vec<String> = legal
                .iter()
                .map(|&m| format!("{}:{}", uci(&pos, m), pos.policy_index(m)))
                .collect();
            println!("{}\t{}", pos.fen(), encoded.join(","));

            let pick = (rng.next() % legal.len() as u64) as usize;
            let mv = legal[pick];
            pos.play(mv);
        }
    }
}

/// UCI in python-chess's convention: castling is the king's two-square move.
fn uci(pos: &ChessPos, mv: Move) -> String {
    let (from, to) = king_dest(pos, mv);
    let promo = match mv {
        Move::Normal {
            promotion: Some(role),
            ..
        } => role.char().to_string(),
        _ => String::new(),
    };
    format!("{from}{to}{promo}")
}

fn king_dest(_pos: &ChessPos, mv: Move) -> (shakmaty::Square, shakmaty::Square) {
    match mv {
        Move::Castle { king, rook } => {
            let to_file = if rook.file() > king.file() {
                shakmaty::File::G
            } else {
                shakmaty::File::C
            };
            (king, shakmaty::Square::from_coords(to_file, king.rank()))
        }
        _ => (mv.from().expect("no drops in standard chess"), mv.to()),
    }
}

fn arg<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    let i = args.iter().position(|a| a == flag)?;
    args.get(i + 1)?.parse().ok()
}

/// xorshift64*, so the dump is reproducible without a dev-dependency.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}
