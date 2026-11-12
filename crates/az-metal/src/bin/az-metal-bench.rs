//! Positions per second through the native Metal path, call by call.
//!
//!     cargo run -p az-metal --release --bin az-metal-bench -- net.azmw [batches] [seconds]
//!
//! Each call is what self-play would make: copy one packed batch in, run the
//! graph synchronously, read legal logits and values back. Inputs are random
//! masks of realistic density with a legal width of 64; the network's cost
//! does not depend on the values.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("az-metal-bench needs macOS (Metal)");
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
fn main() {
    use az_metal::{MetalNet, Precision, Weights};
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: az-metal-bench net.azmw [128,256,...] [seconds]");
    let batches: Vec<usize> = args
        .get(2)
        .map(|s| s.split(',').map(|v| v.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![128, 256, 512, 1024, 2048]);
    let secs: f64 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(3.0);
    let width = 64;

    let w = Weights::load(path).unwrap();
    let a = w.arch;
    println!(
        "net {}x{} in_planes {} policy {}  legal width {width}  {secs} s per point",
        a.blocks, a.channels, a.in_planes, a.policy_len
    );

    let mut state = 0x9E3779B97F4A7C15u64;
    let mut rnd = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for prec in [Precision::F32, Precision::F16] {
        let mut net = MetalNet::new(w.clone(), prec).unwrap();
        println!("\n{prec:?} on {}", net.device_name());
        println!("{:>7} {:>11} {:>10} {:>8}", "batch", "pos/sec", "ms/call", "calls");
        for &n in &batches {
            // Sparse masks (a few pieces per plane) and unit scales.
            let bits: Vec<u64> = (0..n * a.in_planes).map(|_| rnd() & rnd() & rnd()).collect();
            let scale = vec![1.0f32; n * a.in_planes];
            let legal: Vec<i32> = (0..n * width).map(|_| (rnd() % a.policy_len as u64) as i32).collect();
            for _ in 0..5 {
                net.forward_packed(n, width, &bits, &scale, &legal, false);
            }
            let t0 = Instant::now();
            let mut calls = 0usize;
            let mut sink = 0.0f32;
            while t0.elapsed().as_secs_f64() < secs {
                let out = net.forward_packed(n, width, &bits, &scale, &legal, false);
                sink += out.values[0];
                calls += 1;
            }
            let dt = t0.elapsed().as_secs_f64();
            println!(
                "{n:>7} {:>11.0} {:>10.3} {calls:>8}{}",
                (calls * n) as f64 / dt,
                dt * 1e3 / calls as f64,
                if sink.is_nan() { "  (NaN!)" } else { "" }
            );
        }
    }
}
