//! What can the self-play driver feed, with the network taken out of the picture?
//!
//! Before renting a bigger GPU it is worth knowing the ceiling on the other side
//! of the boundary. This runs the real chess self-play driver -- tree descent,
//! expansion, backup, and observation encoding -- against an instant evaluator,
//! so the number it reports is the rate at which a *single thread* of Rust can
//! supply positions.
//!
//! If that ceiling sits below what a rented accelerator can consume, the
//! accelerator idles and the money is wasted. The fix is threading, not hardware.

use az_core::chess::ChessPos;
use az_core::game::{obs_len, Game};
use az_core::mcts::Config;
use az_core::selfplay::SelfPlay;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sims: u32 = flag(&args, "--sims").unwrap_or(32);
    let concurrency: usize = flag(&args, "--concurrency").unwrap_or(512);
    let games: usize = flag(&args, "--games").unwrap_or(512);

    println!(
        "Self-play driver ceiling, single-threaded, network excluded\n\
         chess  sims {sims}  concurrency {concurrency}  games {games}\n"
    );

    let cfg = Config {
        sims,
        max_considered: 16,
        ..Config::default()
    };
    let mut sp = SelfPlay::<ChessPos>::new(concurrency, games, cfg, 300, 0xC0FFEE);

    let obs = obs_len::<ChessPos>();
    let mut buf = Vec::with_capacity(concurrency * obs);
    // A flat evaluator: the point is to measure everything except the network.
    let logits = vec![0.0f32; concurrency * ChessPos::POLICY_LEN];
    let values = vec![0.0f32; concurrency];

    let start = Instant::now();
    let mut evals: u64 = 0;
    let mut batches: u64 = 0;

    loop {
        let n = sp.next_batch(&mut buf);
        if n == 0 {
            break;
        }
        sp.submit(&logits[..n * ChessPos::POLICY_LEN], &values[..n]);
        evals += n as u64;
        batches += 1;
    }

    let secs = start.elapsed().as_secs_f64();
    let (first, second, draws, mean_plies) = sp.stats();

    println!("{:<26} {:>14}", "games completed", sp.games_completed());
    println!("{:<26} {:>14}", "leaf evaluations", evals);
    println!("{:<26} {:>14}", "batches", batches);
    println!(
        "{:<26} {:>14.1}",
        "mean batch size",
        evals as f64 / batches as f64
    );
    println!("{:<26} {:>14.2}", "seconds", secs);
    println!();
    println!(
        "{:<26} {:>14.0}",
        "evals/sec (one thread)",
        evals as f64 / secs
    );
    println!(
        "{:<26} {:>14.2}",
        "games/sec (one thread)",
        sp.games_completed() as f64 / secs
    );
    println!("{:<26} {:>14.1}", "us per eval", secs * 1e6 / evals as f64);
    println!();
    println!(
        "  results: {first} / {second} / {draws} (first/second/draw), mean {mean_plies:.1} plies"
    );
    println!(
        "  encoding alone moves {:.1} MB/s of planes",
        (evals as f64 * obs as f64 * 4.0) / secs / 1e6
    );
}

fn flag<T: std::str::FromStr>(args: &[String], name: &str) -> Option<T> {
    let i = args.iter().position(|a| a == name)?;
    args.get(i + 1)?.parse().ok()
}
