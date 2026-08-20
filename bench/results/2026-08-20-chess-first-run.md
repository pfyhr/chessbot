# Phase 3: the loop on chess

**Date:** 2026-08-20 · **Machine:** Apple M2 Max
**Run:** 25 generations, 256 games each, 32 sims/move, 6x96 net (11.0M params)
**Cost:** 6,400 games, 35.6M evaluations, **61.7 minutes**

```
.venv/bin/chess-train --generations 25 --games 256 --concurrency 256 --out runs/chess-v1
.venv/bin/chess-report runs/chess-v1 --games 200 --every 2
```

## Result: it learns

| gen | material vs random | defend | mate | opening |
|---:|---:|---:|---:|---:|
| 0 | **-1.31** | 0.210 | 0.030 | 0.145 |
| 6 | +0.49 | 0.230 | 0.020 | 0.511 |
| 12 | +1.12 | 0.290 | 0.015 | 0.202 |
| 18 | +2.72 | 0.280 | 0.010 | 0.111 |
| 24 | **+4.04** | 0.285 | 0.010 | 0.332 |
| *random baseline* | *-0.06* | *0.229* | *0.031* | *0.200* |

Last generation against first, 200 games, randomised openings:

| condition | result | score | Elo |
|---|---|---:|---:|
| raw policy | 2W 2L **196D** | 0.500 | +0 |
| search @32 | **94W 24L 82D** | 0.675 | **+127** |

## Reading it

**Material is the signal.** It goes from -1.31 to +4.04 against a metric calibrated
at -0.06 for random-against-random. The network stops hanging pieces and starts
winning them. This is the honest evidence that the loop learns chess, and it was
invisible to every metric the run started with.

**Search is what makes the improvement visible.** The raw-policy head-to-head is
196 draws out of 200 — neither network can convert an advantage without search, so
every game hits the ply cap. Add 32 simulations and the same two networks separate
by 127 Elo. This is the mirror image of Connect4, where the *raw* margin was the
larger one: there both nets could finish games, so the policy difference dominated.
In chess at this strength, search is doing the converting.

**Two metrics were dead on arrival and are worth naming.** `vs-prev` returned
exactly 0.500 in twenty-two of twenty-five generations — that is 60 draws out of 60,
every time, not a coincidence. Raw-policy match play cannot distinguish networks
this weak. Anyone reading the training table alone would conclude nothing was
happening.

## What is not working

**Mate-in-1 never leaves chance** (0.030 → 0.010 against a 0.031 baseline). The
suite is generated from random walks, which produce positions with both kings
marooned in the open — exact ground truth, but a hard out-of-distribution probe.
The network has no early reason to be good at it, and 6,400 games is not enough to
develop tactics that transfer to positions it never sees.

**Opening preference oscillates violently**: 0.145 → 0.596 → 0.111 → 0.730 → 0.332.
It is not drifting, it is swinging, which points at the training setup rather than
at slow learning. The replay window is four generations, so the network chases
recent data; a longer window and a lower learning rate are the first two things to
try.

## Scale, in perspective

6,400 games is roughly **0.015%** of AlphaZero's 44 million. The result here is that
the machinery works end to end on chess and produces a measurable, search-verified
improvement in an hour — not that the engine is any good. It is not: it plays
Alekhine's Defence and then loses a rook.

## What the run cost per hour

35.6M evaluations in 61.7 minutes is ~9,600 evals/sec sustained, against 23,678 for
inference alone. The difference is training and evaluation sharing the same GPU,
which is exactly the caveat attached to the throughput projections.
