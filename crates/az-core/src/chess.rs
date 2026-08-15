//! Chess via `shakmaty`, plus the AlphaZero position and move encodings.
//!
//! # The move encoding
//!
//! AlphaZero encodes a move as one of 73 "planes" attached to the move's origin
//! square, giving a flat policy vector of 73 * 64 = 4672:
//!
//! * **0..56  — queen moves.** 8 directions x 7 distances. Covers every king,
//!   queen, rook, bishop and ordinary pawn move, plus en passant and castling.
//! * **56..64 — knight moves.** One plane per knight delta.
//! * **64..73 — underpromotions.** 3 pieces (knight, bishop, rook) x 3 directions
//!   (capture left, straight, capture right). Promoting to a *queen* is not here;
//!   it is encoded as the plain pawn push/capture it rides on.
//!
//! The flat index is `plane * 64 + from_square`, which is plane-major and so
//! matches a `(73, 8, 8)` convolutional policy head flattened in PyTorch's
//! `(C, H, W)` order. Getting this order wrong is silent: the network trains
//! happily against transposed targets and simply never learns.
//!
//! # Two traps worth naming
//!
//! 1. `shakmaty::Move::to()` returns the **rook's** square for a castling move,
//!    not the king's destination. AlphaZero encodes castling as the king moving
//!    two squares, so [`ChessPos::policy_index`] reconstructs the king target
//!    rather than trusting `to()`.
//!
//! 2. A castle and a queen move can produce the *same* index — a king castling
//!    e1->g1 is "east, two squares from e1", exactly like a queen on e1 going to
//!    g1. This is harmless because they cannot both be legal in one position:
//!    castling requires the king on e1, which excludes a queen being there. The
//!    encoding only has to be injective over the legal moves of a *single*
//!    position, and it is.
//!
//! # Perspective
//!
//! Everything is encoded from the mover's point of view: when Black is to move
//! the board is mirrored vertically, so the side to move always advances up the
//! board. Move encodings are mirrored to match.

use std::sync::Arc;

use shakmaty::zobrist::Zobrist64;
use shakmaty::{
    Board, CastlingMode, CastlingSide, Chess, Color, EnPassantMode, File, Move, Piece, Position,
    Role, Square,
};

use crate::game::{Game, Outcome, Player};

/// How many past positions feed the observation stack.
///
/// AlphaZero used 8. History mainly buys repetition awareness, which we also
/// encode explicitly, so smaller values are worth measuring — this is a single
/// knob and [`ChessPos::OBS_SHAPE`] follows it automatically.
pub const HISTORY: usize = 8;

/// Planes per history step: 6 own pieces + 6 opponent pieces + 2 repetition.
const PLANES_PER_STEP: usize = 14;
/// Side to move, move number, 4 castling rights, halfmove clock.
const META_PLANES: usize = 7;
const TOTAL_PLANES: usize = HISTORY * PLANES_PER_STEP + META_PLANES;

const QUEEN_PLANES: usize = 56;
const KNIGHT_BASE: usize = QUEEN_PLANES;
const UNDERPROMO_BASE: usize = 64;
const NUM_PLANES: usize = 73;

/// Queen-move directions as (file delta, rank delta), in plane order.
const DIRECTIONS: [(i32, i32); 8] = [
    (0, 1),   // N
    (1, 1),   // NE
    (1, 0),   // E
    (1, -1),  // SE
    (0, -1),  // S
    (-1, -1), // SW
    (-1, 0),  // W
    (-1, 1),  // NW
];

/// Knight deltas as (file delta, rank delta), in plane order.
const KNIGHT_DELTAS: [(i32, i32); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];

/// The six roles in observation-plane order.
const ROLES: [Role; 6] = [
    Role::Pawn,
    Role::Knight,
    Role::Bishop,
    Role::Rook,
    Role::Queen,
    Role::King,
];

/// One past position, kept for repetition detection and the observation stack.
///
/// Linked through `Arc` rather than copied into every position: MCTS clones
/// positions once per node visit, and copying eight boards each time would swamp
/// the make-move cost we chose `shakmaty` for. Sibling branches of the search
/// tree share their common ancestry instead.
struct HistNode {
    board: Board,
    hash: Zobrist64,
    parent: Option<Arc<HistNode>>,
}

/// A chess position together with enough history to detect repetitions.
#[derive(Clone)]
pub struct ChessPos {
    pos: Chess,
    /// Most recent first, and includes the current position.
    hist: Option<Arc<HistNode>>,
}

impl ChessPos {
    pub fn new() -> Self {
        Self::from_position(Chess::default())
    }

    pub fn from_position(pos: Chess) -> Self {
        let head = HistNode {
            board: pos.board().clone(),
            hash: pos.zobrist_hash(EnPassantMode::Legal),
            parent: None,
        };
        Self {
            pos,
            hist: Some(Arc::new(head)),
        }
    }

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let parsed: shakmaty::fen::Fen = fen.parse().map_err(|e| format!("bad FEN: {e}"))?;
        let pos: Chess = parsed
            .into_position(CastlingMode::Standard)
            .map_err(|e| format!("illegal position: {e}"))?;
        Ok(Self::from_position(pos))
    }

    pub fn fen(&self) -> String {
        shakmaty::fen::Fen::from_position(&self.pos, EnPassantMode::Always).to_string()
    }

    pub fn inner(&self) -> &Chess {
        &self.pos
    }

    /// How many times the current position has occurred, including now.
    ///
    /// Only positions since the last irreversible move can repeat, and the
    /// halfmove clock counts exactly those, so it bounds the walk.
    pub fn repetition_count(&self) -> usize {
        let Some(head) = &self.hist else {
            return 0;
        };
        let target = head.hash;
        let window = self.pos.halfmoves() as usize;

        let mut count = 0;
        let mut node = Some(head);
        let mut steps = 0;
        while let Some(n) = node {
            if n.hash == target {
                count += 1;
            }
            if steps >= window {
                break;
            }
            steps += 1;
            node = n.parent.as_ref();
        }
        count
    }

    /// Iterate history most-recent-first, current position included.
    fn history_iter(&self) -> impl Iterator<Item = &HistNode> {
        let mut node = self.hist.as_deref();
        std::iter::from_fn(move || {
            let cur = node?;
            node = cur.parent.as_deref();
            Some(cur)
        })
    }

    /// Map a square into the mover's frame, mirroring vertically for Black.
    #[inline]
    fn rel(sq: Square, flip: bool) -> Square {
        if flip {
            sq.flip_vertical()
        } else {
            sq
        }
    }

    /// Origin and true destination of a move, castling reconstructed.
    fn from_to(mv: Move) -> (Square, Square) {
        match mv {
            Move::Castle { king, rook } => {
                // `Move::to()` would hand back the rook square here.
                let side = if rook.file() > king.file() {
                    CastlingSide::KingSide
                } else {
                    CastlingSide::QueenSide
                };
                let to_file = match side {
                    CastlingSide::KingSide => File::G,
                    CastlingSide::QueenSide => File::C,
                };
                (king, Square::from_coords(to_file, king.rank()))
            }
            _ => (
                mv.from().expect("drops do not occur in standard chess"),
                mv.to(),
            ),
        }
    }
}

impl Default for ChessPos {
    fn default() -> Self {
        Self::new()
    }
}

/// Plane index for a queen-style move, or `None` if it is not a straight line.
fn queen_plane(df: i32, dr: i32) -> Option<usize> {
    if df != 0 && dr != 0 && df.abs() != dr.abs() {
        return None;
    }
    let dist = df.abs().max(dr.abs());
    if dist == 0 || dist > 7 {
        return None;
    }
    let dir = DIRECTIONS
        .iter()
        .position(|&(f, r)| f == df.signum() && r == dr.signum())?;
    Some(dir * 7 + (dist as usize - 1))
}

fn knight_plane(df: i32, dr: i32) -> Option<usize> {
    KNIGHT_DELTAS
        .iter()
        .position(|&(f, r)| f == df && r == dr)
        .map(|i| KNIGHT_BASE + i)
}

fn underpromo_plane(role: Role, df: i32) -> Option<usize> {
    let piece = match role {
        Role::Knight => 0,
        Role::Bishop => 1,
        Role::Rook => 2,
        _ => return None,
    };
    // df is -1, 0 or +1 in the mover's frame: capture left, straight, capture right.
    let dir = usize::try_from(df + 1).ok()?;
    if dir > 2 {
        return None;
    }
    Some(UNDERPROMO_BASE + piece * 3 + dir)
}

impl Game for ChessPos {
    type Move = Move;

    const POLICY_LEN: usize = NUM_PLANES * 64;
    const OBS_SHAPE: (usize, usize, usize) = (TOTAL_PLANES, 8, 8);

    fn initial() -> Self {
        Self::new()
    }

    fn player_to_move(&self) -> Player {
        match self.pos.turn() {
            Color::White => Player::First,
            Color::Black => Player::Second,
        }
    }

    fn legal_moves(&self, out: &mut Vec<Self::Move>) {
        out.clear();
        out.extend(self.pos.legal_moves().iter().copied());
    }

    fn play(&mut self, mv: Self::Move) {
        self.pos.play_unchecked(mv);
        let node = HistNode {
            board: self.pos.board().clone(),
            hash: self.pos.zobrist_hash(EnPassantMode::Legal),
            parent: self.hist.take(),
        };
        self.hist = Some(Arc::new(node));
    }

    fn outcome(&self) -> Option<Outcome> {
        if self.pos.is_checkmate() {
            // The side to move has been mated.
            return Some(Outcome::Loss);
        }
        if self.pos.is_stalemate() || self.pos.is_insufficient_material() {
            return Some(Outcome::Draw);
        }
        // Claimed automatically, as in AlphaZero: no engine declines these.
        if self.pos.halfmoves() >= 100 || self.repetition_count() >= 3 {
            return Some(Outcome::Draw);
        }
        None
    }

    fn policy_index(&self, mv: Self::Move) -> usize {
        let flip = self.pos.turn() == Color::Black;
        let (raw_from, raw_to) = Self::from_to(mv);
        let from = Self::rel(raw_from, flip);
        let to = Self::rel(raw_to, flip);

        let df = i32::from(to.file()) - i32::from(from.file());
        let dr = i32::from(to.rank()) - i32::from(from.rank());

        let plane = if let Move::Normal {
            promotion: Some(role),
            ..
        } = mv
        {
            // Queen promotions ride on the pawn move's own plane; only the
            // lesser pieces get dedicated planes.
            if role == Role::Queen {
                queen_plane(df, dr).expect("queen promotion is a one-square pawn move")
            } else {
                underpromo_plane(role, df).expect("underpromotion must be knight, bishop or rook")
            }
        } else if let Some(p) = knight_plane(df, dr) {
            p
        } else {
            queen_plane(df, dr).expect("move is neither a knight hop nor a straight line")
        };

        plane * 64 + usize::from(from)
    }

    fn move_from_policy_index(&self, index: usize) -> Option<Self::Move> {
        // Decode by re-encoding: the inverse is correct by construction, and any
        // drift between the two directions becomes impossible rather than merely
        // unlikely. MCTS always has the legal move list to hand when it needs
        // this, so the linear scan is not on a hot path.
        self.pos
            .legal_moves()
            .iter()
            .copied()
            .find(|&mv| self.policy_index(mv) == index)
    }

    fn encode(&self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), TOTAL_PLANES * 64);
        out.fill(0.0);

        let us = self.pos.turn();
        let them = !us;
        let flip = us == Color::Black;

        // --- history stack -------------------------------------------------
        // Repetition planes describe the current position only; earlier steps
        // carry piece placement. That is enough for the network to see a
        // repetition looming without us threading a count through every node.
        let reps = self.repetition_count();
        for (step, node) in self.history_iter().take(HISTORY).enumerate() {
            let base = step * PLANES_PER_STEP;
            for (i, &role) in ROLES.iter().enumerate() {
                for (offset, color) in [(0, us), (6, them)] {
                    let bb = node.board.by_piece(Piece { color, role });
                    let plane = base + offset + i;
                    for sq in bb {
                        out[plane * 64 + usize::from(Self::rel(sq, flip))] = 1.0;
                    }
                }
            }
            if step == 0 {
                if reps >= 2 {
                    fill_plane(out, base + 12);
                }
                if reps >= 3 {
                    fill_plane(out, base + 13);
                }
            }
        }

        // --- metadata ------------------------------------------------------
        let meta = HISTORY * PLANES_PER_STEP;
        if us == Color::Black {
            fill_plane(out, meta);
        }
        // Normalised rather than raw so the scale matches the binary planes.
        fill_plane_with(out, meta + 1, self.pos.fullmoves().get() as f32 / 100.0);

        let castles = self.pos.castles();
        for (i, (color, side)) in [
            (us, CastlingSide::KingSide),
            (us, CastlingSide::QueenSide),
            (them, CastlingSide::KingSide),
            (them, CastlingSide::QueenSide),
        ]
        .into_iter()
        .enumerate()
        {
            if castles.has(color, side) {
                fill_plane(out, meta + 2 + i);
            }
        }

        fill_plane_with(out, meta + 6, self.pos.halfmoves() as f32 / 100.0);
    }
}

#[inline]
fn fill_plane(out: &mut [f32], plane: usize) {
    fill_plane_with(out, plane, 1.0);
}

#[inline]
fn fill_plane_with(out: &mut [f32], plane: usize, value: f32) {
    out[plane * 64..(plane + 1) * 64].fill(value);
}

impl From<ChessPos> for Chess {
    fn from(p: ChessPos) -> Chess {
        p.pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::obs_len;

    fn moves_of(p: &ChessPos) -> Vec<Move> {
        let mut v = Vec::new();
        p.legal_moves(&mut v);
        v
    }

    #[test]
    fn shapes_are_what_alphazero_specifies() {
        assert_eq!(ChessPos::POLICY_LEN, 4672);
        assert_eq!(ChessPos::OBS_SHAPE, (119, 8, 8));
        assert_eq!(obs_len::<ChessPos>(), 119 * 64);
    }

    /// The property the whole training pipeline rests on: within any single
    /// position, distinct legal moves must get distinct policy indices.
    #[test]
    fn policy_index_is_injective_over_a_random_walk() {
        let mut seen_positions = 0;
        // A cheap deterministic PRNG; no dev-dependency needed for this.
        let mut rng: u64 = 0x9E3779B97F4A7C15;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };

        for game in 0..200 {
            let mut pos = ChessPos::new();
            for _ in 0..120 {
                if pos.outcome().is_some() {
                    break;
                }
                let legal = moves_of(&pos);
                if legal.is_empty() {
                    break;
                }

                let mut indices: Vec<usize> = legal.iter().map(|&m| pos.policy_index(m)).collect();
                let before = indices.len();
                indices.sort_unstable();
                indices.dedup();
                assert_eq!(
                    indices.len(),
                    before,
                    "collision in game {game} at FEN {}",
                    pos.fen()
                );

                for &i in &indices {
                    assert!(i < ChessPos::POLICY_LEN, "index {i} out of range");
                }

                // Round-trip every legal move through the flat index.
                for &m in &legal {
                    let idx = pos.policy_index(m);
                    assert_eq!(
                        pos.move_from_policy_index(idx),
                        Some(m),
                        "round-trip failed for {m:?} at FEN {}",
                        pos.fen()
                    );
                }

                seen_positions += 1;
                let pick = (next() % legal.len() as u64) as usize;
                pos.play(legal[pick]);
            }
        }
        assert!(seen_positions > 1000, "walk was too short to mean much");
    }

    /// Castling must encode as the king moving two squares, not as a move to the
    /// rook's square.
    #[test]
    fn castling_encodes_as_a_two_square_king_move() {
        let pos = ChessPos::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        let legal = moves_of(&pos);
        let castles: Vec<Move> = legal
            .iter()
            .copied()
            .filter(|m| matches!(m, Move::Castle { .. }))
            .collect();
        assert_eq!(castles.len(), 2, "kiwipete has both castles available");

        for c in castles {
            let (from, to) = ChessPos::from_to(c);
            assert_eq!(from, Square::E1);
            assert!(to == Square::G1 || to == Square::C1, "got {to:?}");

            // East/west, two squares: planes 14..21 (E) and 42..49 (W), distance 2.
            let plane = pos.policy_index(c) / 64;
            let expected = if to == Square::G1 {
                2 * 7 + 1 // east, distance 2
            } else {
                6 * 7 + 1 // west, distance 2
            };
            assert_eq!(plane, expected, "castle to {to:?} landed on plane {plane}");
        }
    }

    /// All four underpromotions from one pawn must be distinguishable, and the
    /// queen promotion must not collide with them.
    #[test]
    fn promotions_occupy_the_right_planes() {
        // White pawn on b7 with b8 empty and enemy pieces on a8 and c8, so all
        // three promotion directions are live: capture left, straight, capture
        // right. Blocking b8 would silently reduce this to two directions and
        // the test would still pass while checking less.
        let pos = ChessPos::from_fen("r1b1k3/1P6/8/8/8/8/8/4K3 w q - 0 1").unwrap();
        let legal = moves_of(&pos);
        let promos: Vec<Move> = legal
            .iter()
            .copied()
            .filter(|m| {
                matches!(
                    m,
                    Move::Normal {
                        promotion: Some(_),
                        ..
                    }
                )
            })
            .collect();
        assert_eq!(
            promos.len(),
            12,
            "expected 3 directions x 4 pieces at {}",
            pos.fen()
        );

        let mut indices: Vec<usize> = promos.iter().map(|&m| pos.policy_index(m)).collect();
        let before = indices.len();
        indices.sort_unstable();
        indices.dedup();
        assert_eq!(indices.len(), before, "promotion indices collided");

        for &m in &promos {
            let plane = pos.policy_index(m) / 64;
            match m {
                Move::Normal {
                    promotion: Some(Role::Queen),
                    ..
                } => {
                    assert!(
                        plane < QUEEN_PLANES,
                        "queen promo should ride a queen plane"
                    );
                }
                _ => {
                    assert!(
                        (UNDERPROMO_BASE..NUM_PLANES).contains(&plane),
                        "underpromotion landed on plane {plane}"
                    );
                }
            }
        }
    }

    /// Black's encoding must be the mirror of White's: the same position with
    /// colours and ranks flipped should produce identical planes and indices.
    #[test]
    fn black_sees_a_mirrored_board() {
        let white = ChessPos::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1").unwrap();
        let black = ChessPos::from_fen("4k3/4p3/8/8/8/8/8/4K3 b - - 0 1").unwrap();

        let mut a = vec![0.0f32; obs_len::<ChessPos>()];
        let mut b = vec![0.0f32; obs_len::<ChessPos>()];
        white.encode(&mut a);
        black.encode(&mut b);

        // The side-to-move plane is the one legitimate difference.
        let meta = HISTORY * PLANES_PER_STEP;
        for p in 0..TOTAL_PLANES {
            if p == meta {
                continue;
            }
            assert_eq!(
                &a[p * 64..(p + 1) * 64],
                &b[p * 64..(p + 1) * 64],
                "plane {p} differs between mirrored positions"
            );
        }

        // e2-e4 for White encodes the same as e7-e5 for Black.
        let wm = moves_of(&white)
            .into_iter()
            .find(|m| m.from() == Some(Square::E2) && m.to() == Square::E4)
            .unwrap();
        let bm = moves_of(&black)
            .into_iter()
            .find(|m| m.from() == Some(Square::E7) && m.to() == Square::E5)
            .unwrap();
        assert_eq!(white.policy_index(wm), black.policy_index(bm));
    }

    #[test]
    fn threefold_repetition_is_a_draw() {
        let mut pos = ChessPos::new();
        // Shuffle knights back and forth: Nf3 Nf6 Ng1 Ng8, twice over.
        let shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"];
        assert_eq!(pos.repetition_count(), 1);
        for round in 0..2 {
            for uci in shuffle {
                let mv = moves_of(&pos)
                    .into_iter()
                    .find(|m| {
                        let (f, t) = ChessPos::from_to(*m);
                        format!("{f}{t}") == uci
                    })
                    .unwrap_or_else(|| panic!("{uci} not legal at {}", pos.fen()));
                pos.play(mv);
            }
            assert_eq!(
                pos.repetition_count(),
                round + 2,
                "after {} shuffles",
                round + 1
            );
        }
        assert_eq!(pos.outcome(), Some(Outcome::Draw));
    }

    #[test]
    fn checkmate_is_a_loss_for_the_side_to_move() {
        // Fool's mate: Black has just played Qh4#, White to move and mated.
        let pos =
            ChessPos::from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
                .unwrap();
        assert_eq!(pos.outcome(), Some(Outcome::Loss));
    }

    #[test]
    fn encode_marks_exactly_the_occupied_squares() {
        let pos = ChessPos::new();
        let mut buf = vec![0.0f32; obs_len::<ChessPos>()];
        pos.encode(&mut buf);

        // Step 0: 16 own pieces and 16 opponent pieces on the board.
        let occupied: f32 = buf[0..12 * 64].iter().sum();
        assert_eq!(occupied, 32.0);

        // Pawns are plane 0 for the mover; from White's view that is rank 2.
        for file in 0..8 {
            assert_eq!(buf[8 + file], 1.0, "missing own pawn on file {file}");
        }
    }
}
