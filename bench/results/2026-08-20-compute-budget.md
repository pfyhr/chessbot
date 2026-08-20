# One M2 Max against 5,000 TPUs

**Date:** 2026-08-20 · **Machine:** Apple M2 Max, PyTorch 2.13 on MPS

```
.venv/bin/python bench/az_scale.py
```

The AlphaZero network was run on this machine directly rather than extrapolated from
FLOP counts.

## Measured

| network | params | throughput |
|---|---:|---:|
| ours, 6x96 | 11.0M | 23,772 pos/sec |
| AlphaZero, 19x256 | 33.5M | 2,190 pos/sec |

## Derived

AlphaZero played 44M games in 9 hours on 5,000 TPUs — 1,358 games/sec in total,
0.272 games/sec per TPU. At 135 plies per game:

| | games/sec | 44M games |
|---|---:|---:|
| AlphaZero, all 5,000 TPUs | 1,358 | 9 hours |
| AlphaZero, one TPU | 0.272 | 5.1 years |
| M2 Max, AlphaZero settings (800 sims, 19x256) | 0.020 | **68.8 years** |
| M2 Max, our settings (32 sims, 6x96) | 5.50 | **93 days** |

## What this says

**The per-chip gap is 13.4x, not 66,898x.** One consumer laptop is within about an
order of magnitude of a single first-generation TPU on this workload. Essentially the
whole deficit is parallelism — they had five thousand chips — rather than anything
about the silicon.

**271x of it is already recovered by algorithm choice on the same hardware:** 25x from
Gumbel search running 32 simulations instead of 800, and 10.9x from sizing the network
to the compute available. That converts 68.8 years into 93 days without touching the
machine.

At ~20,000 games/hour, a day of self-play is 475,000 games and a week is 3.3 million.
So the binding constraint is no longer games-per-hour, which is now measured — it is
games-to-target, which is not.

## Caveats

- Inference only. Real training competes for the same GPU, so sustained self-play will
  be lower.
- 135 plies per game is an estimate; untrained weights on zeroed input, which is fine
  for timing convolutions and says nothing about convergence.
- The 5,000-TPU figure is for self-play generation; AlphaZero also used 64 second-
  generation TPUs for training, which is not counted here.
- TPUv1 is an int8 inference part. This compares achieved throughput on the workload,
  not peak specs.
