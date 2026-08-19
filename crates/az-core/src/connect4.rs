//! Connect4 — the correctness oracle for the whole reinforcement-learning loop.
//!
//! Chess is a terrible first target for an AlphaZero implementation, not because
//! it is hard but because it fails *quietly*. A transposed policy target, a
//! sign error in backup, a value learned from the wrong side's perspective — all
//! of these look exactly like "chess is slow to learn" for weeks.
//!
//! Connect4 is small enough that a correct loop converges to near-perfect play in
//! minutes, and its optimal opening is known independently (centre column wins).
//! So it turns a silent failure into a loud one.
//!
//! # Representation
//!
//! The classic position/mask bitboard. The 6x7 grid is stored in 49 bits, seven
//! bits per column: six playable rows plus a sentinel row that is always zero.
//! The sentinel is what makes the win detection safe — without it a vertical
//! shift would carry between columns.
//!
//! * `position` — stones belonging to the player *to move*
//! * `mask` — every stone on the board
//!
//! Playing flips perspective and drops a stone in one step, so there is no turn
//! field to keep in sync.

use std::fmt;

use crate::game::{Game, Outcome, Player};

pub const WIDTH: usize = 7;
pub const HEIGHT: usize = 6;
/// Bits per column: `HEIGHT` playable plus one sentinel.
const COL_BITS: usize = HEIGHT + 1;

/// Own stones and opponent stones.
const PLANES: usize = 2;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Connect4 {
    position: u64,
    mask: u64,
    plies: u32,
}

#[inline]
const fn bottom_bit(col: usize) -> u64 {
    1u64 << (col * COL_BITS)
}

/// The topmost *playable* cell of a column; if it is occupied the column is full.
#[inline]
const fn top_bit(col: usize) -> u64 {
    1u64 << (col * COL_BITS + HEIGHT - 1)
}

/// True if `bb` contains four in a row in any direction.
///
/// Shifts are 1 (vertical), COL_BITS (horizontal) and COL_BITS±1 (the two
/// diagonals). Pairing then pairing again finds four in two steps rather than
/// three.
#[inline]
fn has_four(bb: u64) -> bool {
    const DIRS: [u32; 4] = [1, COL_BITS as u32, COL_BITS as u32 - 1, COL_BITS as u32 + 1];
    for d in DIRS {
        let pairs = bb & (bb >> d);
        if pairs & (pairs >> (2 * d)) != 0 {
            return true;
        }
    }
    false
}

impl Connect4 {
    pub fn new() -> Self {
        Self {
            position: 0,
            mask: 0,
            plies: 0,
        }
    }

    #[inline]
    pub fn can_play(&self, col: usize) -> bool {
        col < WIDTH && (self.mask & top_bit(col)) == 0
    }

    /// Stones of the player who just moved.
    #[inline]
    fn opponent(&self) -> u64 {
        self.position ^ self.mask
    }

    /// Would playing `col` complete a four for the side to move?
    ///
    /// Used by the tactical test suite, not by search.
    pub fn is_winning_move(&self, col: usize) -> bool {
        if !self.can_play(col) {
            return false;
        }
        let landing = (self.mask + bottom_bit(col)) & !self.mask;
        has_four(self.position | landing)
    }

    /// True if the player who just moved has four in a row.
    #[inline]
    pub fn opponent_has_won(&self) -> bool {
        has_four(self.opponent())
    }

    pub fn plies(&self) -> u32 {
        self.plies
    }

    pub fn is_full(&self) -> bool {
        self.plies as usize >= WIDTH * HEIGHT
    }

    /// `'x'` for the side to move, `'o'` for the opponent, `'.'` for empty.
    ///
    /// Row 0 is the bottom of the board, matching the bit layout.
    pub fn cell(&self, row: usize, col: usize) -> char {
        let bit = 1u64 << (col * COL_BITS + row);
        if self.position & bit != 0 {
            'x'
        } else if self.mask & bit != 0 {
            'o'
        } else {
            '.'
        }
    }

    /// Build a position by playing a sequence of columns, 0-indexed.
    ///
    /// Returns `None` if any move is illegal or the game ends early.
    pub fn from_moves(cols: &[usize]) -> Option<Self> {
        let mut pos = Self::new();
        for &c in cols {
            if !pos.can_play(c) || pos.opponent_has_won() {
                return None;
            }
            pos.drop_in(c);
        }
        Some(pos)
    }

    #[inline]
    fn drop_in(&mut self, col: usize) {
        // Flip perspective, then add the stone. `mask + bottom_bit` carries up
        // the column to exactly the lowest empty cell.
        self.position ^= self.mask;
        self.mask |= self.mask + bottom_bit(col);
        self.plies += 1;
    }
}

impl Default for Connect4 {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Connect4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        for row in (0..HEIGHT).rev() {
            for col in 0..WIDTH {
                write!(f, "{} ", self.cell(row, col))?;
            }
            writeln!(f)?;
        }
        for col in 0..WIDTH {
            write!(f, "{col} ")?;
        }
        Ok(())
    }
}

impl Game for Connect4 {
    type Move = u8;

    const POLICY_LEN: usize = WIDTH;
    const OBS_SHAPE: (usize, usize, usize) = (PLANES, HEIGHT, WIDTH);

    fn initial() -> Self {
        Self::new()
    }

    fn player_to_move(&self) -> Player {
        if self.plies.is_multiple_of(2) {
            Player::First
        } else {
            Player::Second
        }
    }

    fn expand(&self, out: &mut Vec<Self::Move>) -> Option<Outcome> {
        out.clear();

        // Check the previous move first: a won position has legal moves and is
        // still over, exactly like a fifty-move draw in chess.
        if self.opponent_has_won() {
            return Some(Outcome::Loss);
        }
        if self.is_full() {
            return Some(Outcome::Draw);
        }

        for col in 0..WIDTH {
            if self.can_play(col) {
                out.push(col as u8);
            }
        }
        debug_assert!(!out.is_empty(), "non-terminal position with no moves");
        None
    }

    fn play(&mut self, mv: Self::Move) {
        debug_assert!(self.can_play(mv as usize), "illegal move {mv}");
        self.drop_in(mv as usize);
    }

    fn policy_index(&self, mv: Self::Move) -> usize {
        mv as usize
    }

    fn move_from_policy_index(&self, index: usize) -> Option<Self::Move> {
        (index < WIDTH && self.can_play(index)).then_some(index as u8)
    }

    fn encode(&self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), PLANES * HEIGHT * WIDTH);
        out.fill(0.0);

        let opponent = self.opponent();
        for col in 0..WIDTH {
            for row in 0..HEIGHT {
                let bit = 1u64 << (col * COL_BITS + row);
                let cell = row * WIDTH + col;
                if self.position & bit != 0 {
                    out[cell] = 1.0;
                } else if opponent & bit != 0 {
                    out[HEIGHT * WIDTH + cell] = 1.0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moves_of(p: &Connect4) -> Vec<u8> {
        let mut v = Vec::new();
        p.legal_moves(&mut v);
        v
    }

    #[test]
    fn opening_has_seven_moves() {
        let p = Connect4::new();
        assert_eq!(moves_of(&p), vec![0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(p.outcome(), None);
        assert_eq!(Connect4::POLICY_LEN, 7);
        assert_eq!(Connect4::OBS_SHAPE, (2, 6, 7));
    }

    #[test]
    fn a_column_fills_after_six_stones() {
        let mut p = Connect4::new();
        // Alternate between column 0 and column 1 so nobody connects four.
        for i in 0..6 {
            p.play(if i % 2 == 0 { 0 } else { 1 });
        }
        // Three stones each in columns 0 and 1.
        assert!(p.can_play(0));
        for i in 0..6 {
            p.play(if i % 2 == 0 { 0 } else { 1 });
        }
        assert!(!p.can_play(0), "column 0 should be full");
        assert!(!p.can_play(1), "column 1 should be full");
        assert!(!moves_of(&p).contains(&0));
    }

    #[test]
    fn vertical_four_wins() {
        // First player stacks column 3; second player answers in column 4.
        let p = Connect4::from_moves(&[3, 4, 3, 4, 3, 4, 3]).unwrap();
        assert!(p.opponent_has_won(), "the stacker just won");
        assert_eq!(p.outcome(), Some(Outcome::Loss));
    }

    #[test]
    fn horizontal_four_wins() {
        let p = Connect4::from_moves(&[0, 0, 1, 1, 2, 2, 3]).unwrap();
        assert_eq!(p.outcome(), Some(Outcome::Loss));
    }

    #[test]
    fn diagonal_four_wins() {
        // Build a rising diagonal for the first player.
        let p = Connect4::from_moves(&[0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3]).unwrap();
        assert_eq!(p.outcome(), Some(Outcome::Loss), "{p:?}");
    }

    #[test]
    fn wins_do_not_wrap_between_columns() {
        // Four stones that are only "in a row" if columns wrap: the top of one
        // column and the bottom of the next. The sentinel row must prevent it.
        let mut p = Connect4::new();
        for _ in 0..6 {
            p.play(0);
        }
        // Column 0 is full and alternating, so nobody has four.
        assert!(!p.opponent_has_won(), "{p:?}");
        assert_eq!(p.outcome(), None);
    }

    #[test]
    fn is_winning_move_agrees_with_playing_it() {
        let mut rng: u64 = 0xDEADBEEF;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };

        for _ in 0..2000 {
            let mut p = Connect4::new();
            loop {
                let legal = moves_of(&p);
                if p.outcome().is_some() || legal.is_empty() {
                    break;
                }
                for &c in &legal {
                    let predicted = p.is_winning_move(c as usize);
                    let mut after = p;
                    after.play(c);
                    assert_eq!(
                        predicted,
                        after.opponent_has_won(),
                        "is_winning_move disagreed for column {c} at {p:?}"
                    );
                }
                let pick = (next() % legal.len() as u64) as usize;
                p.play(legal[pick]);
            }
        }
    }

    #[test]
    fn a_full_board_is_a_draw() {
        // Fill the board in an order that avoids any four in a row.
        let order = [
            0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 2, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 2, 4, 5, 4, 5, 4,
            5, 5, 4, 5, 4, 5, 4, 6, 6, 6, 6, 6, 6,
        ];
        let mut p = Connect4::new();
        for &c in &order {
            if p.outcome().is_some() {
                break;
            }
            p.play(c);
        }
        assert_eq!(p.plies(), 42, "{p:?}");
        assert!(p.is_full());
        // Either a draw, or somebody connected four along the way -- but the
        // board must be full and the game must be over.
        assert!(p.outcome().is_some(), "{p:?}");
    }

    #[test]
    fn encoding_marks_the_right_cells() {
        let p = Connect4::from_moves(&[3]).unwrap();
        let mut buf = vec![0.0f32; 2 * HEIGHT * WIDTH];
        p.encode(&mut buf);

        // One stone on the board, and it belongs to the opponent now that the
        // perspective has flipped.
        assert_eq!(
            buf[..HEIGHT * WIDTH].iter().sum::<f32>(),
            0.0,
            "mover has none"
        );
        assert_eq!(buf[HEIGHT * WIDTH..].iter().sum::<f32>(), 1.0);
        // Bottom row (row 0), column 3.
        assert_eq!(buf[HEIGHT * WIDTH + 3], 1.0);
    }

    #[test]
    fn encoding_is_from_the_movers_perspective() {
        let p = Connect4::from_moves(&[3, 4]).unwrap();
        let mut buf = vec![0.0f32; 2 * HEIGHT * WIDTH];
        p.encode(&mut buf);
        // Two stones, one each.
        assert_eq!(buf[..HEIGHT * WIDTH].iter().sum::<f32>(), 1.0);
        assert_eq!(buf[HEIGHT * WIDTH..].iter().sum::<f32>(), 1.0);
        // The mover is the one who played column 3.
        assert_eq!(buf[3], 1.0, "mover's stone in column 3");
        assert_eq!(buf[HEIGHT * WIDTH + 4], 1.0, "opponent's stone in column 4");
    }

    #[test]
    fn policy_index_round_trips() {
        let p = Connect4::new();
        for c in 0..WIDTH {
            let mv = c as u8;
            assert_eq!(p.policy_index(mv), c);
            assert_eq!(p.move_from_policy_index(c), Some(mv));
        }
        // A full column has no move at its index.
        let mut q = Connect4::new();
        for _ in 0..6 {
            q.play(0);
        }
        assert_eq!(q.move_from_policy_index(0), None);
    }
}
