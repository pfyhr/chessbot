//! Measures the cost of one MCTS node expansion.
//!
//! MCTS does the same two things at every node it visits: find the legal moves,
//! and find out whether the game has already ended. How those two are spelled
//! decides how much movegen work the search does, and movegen is the thing we
//! chose the rules library for.
//!
//! The naive spelling calls `legal_moves()` and then `outcome()`. But shakmaty's
//! `is_checkmate()` and `is_stalemate()` each generate the full legal move list
//! internally, so that pair costs up to *three* movegens per node when one would
//! do. This binary measures the gap so the decision is made on a number.

use az_core::chess::ChessPos;
use az_core::game::Outcome;
use az_core::Game;
use shakmaty::{Move, Position};
use std::time::Instant;

/// The two-call spelling: ask for the moves, then ask whether it is over.
///
/// This is what the `Game` trait looked like before the two were fused, and what
/// the obvious implementation of MCTS would reach for.
fn naive(pos: &ChessPos, buf: &mut Vec<Move>) -> (usize, Option<Outcome>) {
    let inner = pos.inner();
    buf.clear();
    buf.extend(inner.legal_moves().iter().copied());

    // `is_checkmate` and `is_stalemate` each regenerate the list we just built:
    // that is the whole cost this benchmark exists to expose.
    let outcome = if inner.is_checkmate() {
        Some(Outcome::Loss)
    } else if inner.is_stalemate()
        || inner.is_insufficient_material()
        || inner.halfmoves() >= 100
        || pos.repetition_count() >= 3
    {
        Some(Outcome::Draw)
    } else {
        None
    };
    (buf.len(), outcome)
}

/// The fused spelling that shipped: `Game::expand`, one movegen.
fn fused(pos: &ChessPos, buf: &mut Vec<Move>) -> (usize, Option<Outcome>) {
    let outcome = pos.expand(buf);
    (buf.len(), outcome)
}

fn main() {
    let positions = collect_positions(4000);
    println!(
        "MCTS node-expansion cost over {} positions from random play\n",
        positions.len()
    );

    let mut buf = Vec::with_capacity(256);

    // Correctness first: the fast path must agree with the slow one everywhere.
    let mut disagreements = 0;
    for p in &positions {
        let (n1, o1) = naive(p, &mut buf);
        let (n2, o2) = fused(p, &mut buf);
        if n1 != n2 || o1 != o2 {
            disagreements += 1;
            if disagreements <= 3 {
                eprintln!(
                    "  disagree at {}: naive=({n1},{o1:?}) fused=({n2},{o2:?})",
                    p.fen()
                );
            }
        }
    }
    if disagreements > 0 {
        eprintln!("\n{disagreements} disagreements -- the fused path is WRONG, not faster");
        std::process::exit(1);
    }
    println!("  agreement: all {} positions match\n", positions.len());

    let reps = 200;
    let t_naive = time(&positions, reps, naive);
    let t_fused = time(&positions, reps, fused);

    let total = positions.len() * reps;
    println!("{:<10} {:>12} {:>14}", "variant", "ns/node", "nodes/sec");
    println!("{}", "-".repeat(38));
    for (name, secs) in [("naive", t_naive), ("fused", t_fused)] {
        println!(
            "{:<10} {:>12.1} {:>14.0}",
            name,
            secs * 1e9 / total as f64,
            total as f64 / secs
        );
    }
    println!("\nfused is {:.2}x the naive spelling", t_naive / t_fused);
}

fn time(
    positions: &[ChessPos],
    reps: usize,
    f: fn(&ChessPos, &mut Vec<Move>) -> (usize, Option<Outcome>),
) -> f64 {
    let mut buf = Vec::with_capacity(256);
    let mut sink = 0usize;
    let start = Instant::now();
    for _ in 0..reps {
        for p in positions {
            let (n, o) = f(p, &mut buf);
            // Keep the optimiser honest.
            sink = sink.wrapping_add(n + o.is_some() as usize);
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    elapsed
}

/// Positions sampled from random play, so the mix includes real endgames and
/// terminal nodes rather than only wide-open middlegames.
fn collect_positions(target: usize) -> Vec<ChessPos> {
    let mut out = Vec::with_capacity(target);
    let mut rng: u64 = 0x9E3779B97F4A7C15;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut buf = Vec::new();

    while out.len() < target {
        let mut pos = ChessPos::new();
        for _ in 0..200 {
            out.push(pos.clone());
            if out.len() >= target || pos.outcome().is_some() {
                break;
            }
            pos.legal_moves(&mut buf);
            if buf.is_empty() {
                break;
            }
            let pick = (next() % buf.len() as u64) as usize;
            pos.play(buf[pick]);
        }
    }
    out
}
