//! Perft harness: verifies rules correctness and measures raw movegen throughput
//! for each candidate backend.
//!
//! Two subcommands:
//!   verify  -- assert node counts match the reference values (correctness gate)
//!   bench   -- time both backends on the same work (the library decision)
//!
//! Always build with --release. Debug builds are 20-50x slower and any number
//! taken from one is meaningless.

mod backends;
mod positions;

use backends::{CozyChess, PerftBackend, Shakmaty};
use positions::{TestPosition, POSITIONS};
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("bench");

    match cmd {
        "verify" => {
            let max_depth = flag_value(&args, "--max-depth").unwrap_or(5);
            let ok = verify(max_depth);
            if !ok {
                std::process::exit(1);
            }
        }
        "bench" => {
            let depth = flag_value(&args, "--depth").unwrap_or(6);
            let repeat = flag_value(&args, "--repeat").unwrap_or(3);
            let position = flag_str(&args, "--position");
            bench(depth, position.as_deref(), repeat);
        }
        "help" | "--help" | "-h" => usage(),
        other => {
            eprintln!("unknown subcommand: {other}\n");
            usage();
            std::process::exit(2);
        }
    }
}

fn usage() {
    println!(
        "az-perft -- chess movegen correctness + throughput\n\
         \n\
         USAGE:\n  \
           az-perft verify [--max-depth N]      verify node counts for all positions\n  \
           az-perft bench  [--depth N] [--position NAME] [--repeat N]\n\
         \n\
         POSITIONS: {}\n",
        POSITIONS
            .iter()
            .map(|p| p.name)
            .collect::<Vec<_>>()
            .join(", ")
    );
}

// ---------------------------------------------------------------------------
// verify
// ---------------------------------------------------------------------------

fn verify(max_depth: u32) -> bool {
    println!("Verifying node counts up to depth {max_depth}\n");
    let mut failures = 0;
    let mut checks = 0;

    for pos in POSITIONS {
        println!("  {} ", pos.name);
        println!("    {}", pos.fen);
        for depth in 1..=max_depth {
            let Some(expected) = pos.expected(depth as usize) else {
                continue;
            };
            let shak = Shakmaty::perft(pos.fen, depth, false);
            let cozy = CozyChess::perft(pos.fen, depth, false);
            checks += 2;

            let shak_ok = shak == expected;
            let cozy_ok = cozy == expected;
            if !shak_ok {
                failures += 1;
            }
            if !cozy_ok {
                failures += 1;
            }

            println!(
                "    depth {depth}: expected {expected:>12}  shakmaty {} {:>12}  cozy-chess {} {:>12}",
                mark(shak_ok),
                shak,
                mark(cozy_ok),
                cozy,
            );
        }
        println!();
    }

    if failures == 0 {
        println!("All {checks} checks passed.");
        true
    } else {
        println!("{failures} of {checks} checks FAILED.");
        false
    }
}

fn mark(ok: bool) -> &'static str {
    if ok {
        "ok"
    } else {
        "XX"
    }
}

// ---------------------------------------------------------------------------
// bench
// ---------------------------------------------------------------------------

struct Timing {
    nodes: u64,
    elapsed: Duration,
}

impl Timing {
    fn mnps(&self) -> f64 {
        self.nodes as f64 / self.elapsed.as_secs_f64() / 1e6
    }
}

/// Run perft `repeat` times and keep the fastest.
///
/// Best-of-N rather than mean: the fastest run is the one least polluted by
/// scheduler preemption and thermal throttling, so it is the most stable estimate
/// of what the code actually costs. On a laptop this matters -- single-shot
/// timings here vary by well over 10%.
fn time_it<B: PerftBackend>(fen: &str, depth: u32, bulk: bool, repeat: u32) -> Timing {
    let mut nodes = 0;
    let mut best = Duration::MAX;
    for _ in 0..repeat.max(1) {
        let start = Instant::now();
        nodes = B::perft(fen, depth, bulk);
        best = best.min(start.elapsed());
    }
    Timing {
        nodes,
        elapsed: best,
    }
}

fn bench(depth: u32, position: Option<&str>, repeat: u32) {
    let selected: Vec<&TestPosition> = match position {
        Some(name) => match positions::by_name(name) {
            Some(p) => vec![p],
            None => {
                eprintln!("unknown position: {name}");
                std::process::exit(2);
            }
        },
        None => POSITIONS.iter().collect(),
    };

    println!("Perft throughput  (release build, single-threaded)");
    println!(
        "  {} v{}   vs   {} v{}\n",
        Shakmaty::NAME,
        Shakmaty::VERSION,
        CozyChess::NAME,
        CozyChess::VERSION
    );
    println!(
        "{:<14} {:>5} {:>6} {:>14} {:>9} {:>8}   backend",
        "position", "depth", "bulk", "nodes", "seconds", "Mnps"
    );
    println!("{}", "-".repeat(84));

    for pos in selected {
        // Skip depths we have no reference count for -- an unverified timing is
        // just a number, not a measurement.
        if pos.expected(depth as usize).is_none() {
            continue;
        }
        let expected = pos.expected(depth as usize).unwrap();

        for bulk in [false, true] {
            let shak = time_it::<Shakmaty>(pos.fen, depth, bulk, repeat);
            let cozy = time_it::<CozyChess>(pos.fen, depth, bulk, repeat);

            for (name, t) in [(Shakmaty::NAME, &shak), (CozyChess::NAME, &cozy)] {
                let flag = if t.nodes == expected {
                    ""
                } else {
                    "  <-- WRONG"
                };
                println!(
                    "{:<14} {:>5} {:>6} {:>14} {:>9.3} {:>8.1}   {}{}",
                    pos.name,
                    depth,
                    if bulk { "yes" } else { "no" },
                    t.nodes,
                    t.elapsed.as_secs_f64(),
                    t.mnps(),
                    name,
                    flag,
                );
            }

            let ratio = cozy.mnps() / shak.mnps();
            println!(
                "{:<14} {:>5} {:>6} {:>14} {:>9} {:>8}   cozy-chess is {:.2}x shakmaty",
                "", "", "", "", "", "", ratio
            );
        }
        println!();
    }
}

// ---------------------------------------------------------------------------
// tiny arg parsing (not worth a clap dependency yet)
// ---------------------------------------------------------------------------

fn flag_str(args: &[String], flag: &str) -> Option<String> {
    let idx = args.iter().position(|a| a == flag)?;
    args.get(idx + 1).cloned()
}

fn flag_value(args: &[String], flag: &str) -> Option<u32> {
    flag_str(args, flag)?.parse().ok()
}

// ---------------------------------------------------------------------------
// tests -- these run in debug, so keep the depths small
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Both backends must agree with the reference counts at shallow depth.
    /// Deep verification is the `verify` subcommand's job (release build).
    #[test]
    fn shallow_perft_matches_reference() {
        for pos in POSITIONS {
            for depth in 1..=3u32 {
                let Some(expected) = pos.expected(depth as usize) else {
                    continue;
                };
                assert_eq!(
                    Shakmaty::perft(pos.fen, depth, false),
                    expected,
                    "shakmaty wrong on {} depth {depth}",
                    pos.name
                );
                assert_eq!(
                    CozyChess::perft(pos.fen, depth, false),
                    expected,
                    "cozy-chess wrong on {} depth {depth}",
                    pos.name
                );
            }
        }
    }

    /// Bulk counting is an optimisation, not a different definition of perft.
    #[test]
    fn bulk_counting_agrees_with_full_expansion() {
        for pos in POSITIONS {
            for depth in 1..=3u32 {
                assert_eq!(
                    Shakmaty::perft(pos.fen, depth, true),
                    Shakmaty::perft(pos.fen, depth, false),
                    "shakmaty bulk mismatch on {} depth {depth}",
                    pos.name
                );
                assert_eq!(
                    CozyChess::perft(pos.fen, depth, true),
                    CozyChess::perft(pos.fen, depth, false),
                    "cozy-chess bulk mismatch on {} depth {depth}",
                    pos.name
                );
            }
        }
    }
}
