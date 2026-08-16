# Bridge throughput, and the batch size it implies

**Date:** 2026-08-16
**Machine:** Apple M2 Max, 96 GB unified memory
**Stack:** rustc 1.97.1, PyO3 0.29, numpy crate 0.29, Python 3.14.3, PyTorch 2.13.0 (MPS)

Reproduce with:

```
.venv/bin/maturin develop --release
.venv/bin/python bench/bridge_throughput.py
```

## Question

The whole architecture rests on crossing the Rust/Python boundary **once per neural-net
batch** rather than once per MCTS node. That is only worth the complexity if the crossing
is cheap relative to the forward pass it wraps. So: measure both sides.

## Result

### `encode_batch` — positions to `(N, C, 8, 8)` float32

| batch | ms/call | µs/pos | positions/sec |
|---:|---:|---:|---:|
| 1 | 0.001 | 0.87 | 1,142,886 |
| 32 | 0.024 | 0.74 | 1,352,094 |
| 128 | 0.082 | 0.64 | 1,553,097 |
| 512 | 0.368 | 0.72 | 1,393,197 |
| 2048 | 2.137 | 1.04 | 958,241 |

Roughly **1–1.5M positions/sec**, essentially flat. The encoding runs with the GIL
released; only argument marshalling is serialised.

### 6x96 SE-ResNet forward on MPS (1.22M params)

| batch | ms/call | µs/pos | positions/sec | bridge share |
|---:|---:|---:|---:|---:|
| 1 | 2.074 | 2073.54 | 482 | 0.0% |
| 8 | 2.202 | 275.25 | 3,633 | 0.3% |
| 32 | 2.876 | 89.89 | 11,125 | 0.8% |
| 128 | 6.113 | 47.76 | 20,940 | 1.3% |
| 512 | 20.975 | 40.97 | 24,411 | 1.7% |
| 1024 | 41.841 | 40.86 | 24,473 | 2.0% |
| 2048 | 83.758 | 40.90 | 24,451 | 2.5% |

## Conclusions

**1. The boundary is not the bottleneck — 0.0% to 2.5% of combined cost.** The design is
validated with room to spare: the encoder could get 40x more expensive before it started
to matter. No need for shared-memory tricks, pinned buffers, or moving inference into Rust
on these grounds.

**2. MPS has a ~2 ms fixed cost per call, and it dominates everything below batch ~128.**
Batch 1 manages 482 positions/sec against 24,411 at batch 512 — a **50x** difference from
batching alone. This is the single most important number here, and it is exactly why
self-play must run many games concurrently and pool their leaf evaluations. A
straightforward one-game-at-a-time MCTS would leave 98% of the GPU idle.

**3. Batch 512 is the target; past that is free but pointless.** Throughput plateaus at
512 (24,411/sec) with 1024 and 2048 indistinguishable. With `K=4` leaves collected per
tree per pass, that means **G ≈ 128 concurrent self-play games**. Larger batches only add
latency and memory.

## Implied self-play ceiling

At 24,400 evaluations/sec, 32 simulations per move under Gumbel, and ~100 plies per game:

```
24,400 / 32  ≈  760 moves/sec  ≈  7.6 games/sec  ≈  27,000 games/hour
```

That is an NN-bound ceiling that ignores tree overhead, batch under-fill near game end, and
training time sharing the GPU. Even at 10% of it we clear the plan's Phase 5 target of
≥500 games/hour by 5x. The target was set conservatively; the real constraint is likely to
be how many games it takes to learn anything, not how fast we can play them.

## Caveats

- The 6x96 net is representative, not final. Bigger nets shift the ratio further in the
  bridge's favour, not against it.
- Untrained weights and zeroed inputs: fine for timing convolutions, but says nothing about
  convergence.
- Inference only. Training will contend for the same GPU, so sustained self-play throughput
  in a real run will be lower than these figures.
