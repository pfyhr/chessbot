# On a 3090 the network is a quarter of "forward"; two pools buy 1.37x

**Date:** 2026-11-12 · aida, GPU 0 (RTX 3090), `taskset -c 0-3,8-11`, torch
2.11.0+cu128, bf16 autocast, random-init 6x96, `sims 32 / considered 32`.
`bench/generation_profile.py`. Single runs, no error bars; the baseline was
re-measured in the same session as each arm.

## Batch size first

| games = concurrency | precision | forward | driver | total | positions / wall s |
|---:|---|---:|---:|---:|---:|
| 256 | default | 54.5 s | 26.4 s | 82.5 s | 526 |
| 256 | bf16 | 70.5 s | 26.6 s | 98.7 s | 440 |
| 1024 | bf16 | 100.4 s | 118.7 s | 224.2 s | 787 |
| 1024 | bf16 + compile | 110.6 s | 125.0 s | 240.9 s | 763 |

The profiler's default of `--games 64` caps every batch at 64 and measured
8,650 pos/s -- slower than the M2 Max. Not a property of the card.

## Driver threads do not scale here either

At 1024 games: driver 118.7 / 104.7 / 94.2 s at 1 / 4 / 7 threads, i.e. 1.26x
from 7 threads. Same ceiling as on the M2 Max (`2026-10-26-driver-threads.md`).

## Two pools on two Python threads

`python/chessbot/pipeline.py`: N independent game pools, one Python thread
each; `submit` now releases the GIL, so one pool's tree work overlaps the
other's forward.

| config | total | positions / wall s | vs baseline |
|---|---:|---:|---:|
| 1 pool, 1024 games | 213.9 s | 765 | -- |
| 2 pools, 1024 games (512 each) | 183.7 s | 857 | 1.12x |
| 2 pools x 3 driver threads, 1024 games | 219.7 s | 835 | 1.09x |
| **2 pools, 2048 games (1024 each)** | 330.0 s | **1,047** | **1.37x** |

Splitting a fixed number of games halves the batch and gives most of the
overlap back. Keep each pool's batch at the size that saturates the card.

Per-pool CUDA streams were tried first and produced non-finite logits (the
driver's sort panicked on them). Pools share the default stream.

## Where "forward" actually goes

Raw forward throughput, synthetic batches (`--sweep`):

| batch | bf16 | bf16 + compile |
|---:|---:|---:|
| 256 | 80,417 | 117,262 |
| 1024 | 204,048 | 281,109 |
| 4096 | 231,889 | 321,991 |

Self-play at batch 1024 reaches **~57,000 pos/s**. Roughly 70% of what the
profiler labels "network forward" is moving data: per batch of 1024, 31 MB of
float32 observations (almost all 0/1) go up and 19 MB of logits over all 4672
moves (of ~35 legal) come back, both through pageable memory with extra copies
on each side.

## Next

Shrink the bridge: observations as packed bits or uint8 expanded on the
device, logits gathered to the legal moves on the device, and pinned buffers
written by Rust directly. Then CUDA graphs at fixed batch shapes.

## Later the same day: 765 -> 201k evals/s

Each row adds one change to the row above. Steady-state rows are the second
of two streaming generations. Single runs throughout. Where a row says
positions/wall-s it includes 20 training steps; evals/s is self-play only.

| change | commit | result |
|---|---|---:|
| baseline: 1 pool, 1024 games, dense | -- | 765 pos/wall-s |
| 2 pools x 1024 games | `9a0a49d` | 1,047 pos/wall-s |
| packed bridge (bits up, legal logits down) | `414163f` | 1,351 pos/wall-s |
| encode packed planes straight from bitboards | `d49208f` | 2,217 pos/wall-s |
| stream: never drain (mean batch 423 -> full) | `b48a514` | 92.5k evals/s (2 pools) |
| + CUDA graphs, 4 pools x 512 | `b48a514` | 114k evals/s |
| 4 pools x 1024 | -- | 136k evals/s |
| + `torch.compile(dynamic=False)` inside the graphs | this commit | 182k evals/s |
| + channels_last | this commit | 184k (noise) |
| **5 pools x 1024** | this commit | **201k evals/s** |

What each step found:

- **Direct bitboard encoding.** The first packed bridge still built 30 KB of
  floats per position and scanned them back into masks. Driver time per
  evaluation went from 21.5 to 6.8 us.
- **Streaming.** A drained generation's tail ran at a shrinking batch: 5.56M
  evaluations in 13,134 calls, a mean of 423 of 1024 slots. In isolation a
  whole packed forward of 1024 positions takes 6.3 ms (163k pos/s, eager),
  so per-call overhead was never the problem.
- **GPU saturation.** At 4 pools x 512 with graphs, nvidia-smi read a median
  of 81% busy (p90 90%). From there the gains came from cheaper kernels
  (compile) and full batches, not from more overlap.
- **Compile with threads.** `torch.compile` called from two pool threads hung
  (25 min at 0% GPU). Compiling and capturing every bucket on the main thread
  before the pools start avoids it. Dynamo's default recompile limit (8) is
  below the bucket count and falls back to eager silently; the profiler
  raises it.

### Not yet true of training

These are profiler numbers. `train_chess.py` has `--stream`, `--graphs` and
`--pools`, but its self-play still runs default precision (no bf16) and its
compile is not the `dynamic=False` + raised-limit configuration measured here.
`--stream` also changes training semantics -- games straddle weight updates --
and has not been checked against the drained loop by head-to-head Elo.
