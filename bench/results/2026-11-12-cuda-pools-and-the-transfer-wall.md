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
