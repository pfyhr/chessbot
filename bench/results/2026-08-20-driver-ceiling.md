# What can the driver feed, with the network removed?

**Date:** 2026-08-20 · **Machine:** Apple M2 Max

```
./target/release/drivercost --sims 32 --concurrency 512 --games 4096
```

Runs the real chess self-play driver — tree descent, expansion, backup, observation
encoding — against an instant evaluator. The result is the rate at which a **single
thread** of Rust can supply positions to a network.

## Measured

| | |
|---|---:|
| evals/sec, one thread | **292,646** |
| games/sec, one thread | 48.1 |
| per eval | 3.4 µs |
| mean batch | 457 of 512 |
| plane throughput | 9.5 GB/s |

## Against what the GPU can consume

| | evals/sec | driver headroom |
|---|---:|---:|
| M2 Max, AlphaZero net (19x256) | 2,190 | 134x |
| M2 Max, our net (6x96) | 23,678 | 12x |
| one rented 4090 (est. 10x M2 Max) | ~237,000 | 1.2x |
| one rented H100 (est. 25x M2 Max) | ~590,000 | 0.5x — **needs 2–3 threads** |

Optimising the driver on this machine would be pure waste: it is already 12–134x
faster than the GPU it feeds. But it becomes the constraint at roughly one H100, and
games are independent, so the fix is threads rather than better single-thread code.
Size it at about one thread per 300k evals/sec of accelerator.

## Precision is not a lever here

fp16 and bf16 were measured on MPS and buy essentially nothing:

| net | fp32 | fp16 | speedup |
|---|---:|---:|---:|
| 19x256, batch 256 | 2,193 | 2,293 | 1.05x |
| 6x96, batch 512 | 23,678 | 24,484 | 1.03x |

At 8x8 spatial dimensions the work is launch- and bandwidth-bound rather than
compute-bound, so the reduced-precision path has nothing to win. Worth knowing before
spending time on it. On CUDA with tensor cores the answer is likely different, which is
a reason to measure on rented hardware before committing to it.

## Utilisation

The 19x256 network is ~2.87 GFLOP per position, so 2,190 pos/sec is **6.3 TFLOPS
achieved** against roughly 13.6 TFLOPS fp32 peak — about 46%. The M2 Max is being used
properly; there is no large local win hiding here.
