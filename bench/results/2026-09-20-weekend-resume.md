# The raw-policy suites fell for 960 generations while the engine gained 137 Elo

**Date:** 2026-09-20 · 200 games, `tc=16+0.3`, `timemargin=300`,
240-position 4-ply opening book, zero losses on time

`runs/weekend` resumed `runs/night/gen248.pt` on 2026-09-18 with two changes:
`--considered 32` (night ran at the default 16) and `--max-plies 400` (night ran at 200).

## Result

| | result | score | Elo |
|---|---|---:|---:|
| weekend-g983 vs night-g248 | 124W 49L 27D | 0.688 | **+137 ±49** |

LOS 100%. Pentanomial [8, 4, 30, 21, 37], pairs ratio 4.83.

## What this does NOT establish

It is tempting to add this to the 816 from the 2026-08-26 anchor and call the network
"~950 Elo above random". That number should not be published. It is a chain of two
matches, and the anchor link is saturated: the random mover scored **1.2% of 200 games**
there, about 2.4 points. The implied gap swings ~240 Elo on whether it scraped 2 or 4
half-points:

| random's points (of 200) | implied gap |
|---:|---:|
| 1.0 | -920 |
| 2.4 (actual) | -766 |
| 4.0 | -676 |

Compounding the stated error bars gives +/-105 at best, and the random side has no stated
error bar at all because it is saturated. The two matches also ran at different time
controls (8+0.15 standalone vs 16+0.3 under training contention).

**+137 +/-49 against night-g248 is the result. Everything beyond that is inference.**

A defensible absolute figure needs an overlapping ladder -- a chain of booked matches where
every rung scores between roughly 15% and 85%, so no link is saturated, with error
propagated along the chain. The checkpoints for it now exist (every 10th generation of this
run, the night run's, and gen000, which the 2026-08-24 report found indistinguishable from
random). That is STATUS item 4 and it has not been done.

## The trap, for the fifth time

Every raw-policy proxy fell over the same 960 generations that produced +137 Elo:

| | gen 0–50 | gen 950–983 | |
|---|---:|---:|---|
| tactics | 0.275 | 0.206 | ↓ 25% |
| defend | 0.470 | 0.385 | ↓ 18% |
| mate | 0.080 | 0.028 | ↓ to chance (0.031) |
| value loss | 0.305 | 0.352 | ↑ |
| material | 29.25 | 31.30 | ↑ |
| vs-random | 0.774 | 0.782 | flat |

Peak tactics, defend and mate all landed at **generation 18–22** — the values inherited
from gen248 — and drifted down for the rest of the run. Read at face value they say the
run should have been aborted around hour two. The engine was gaining strength the entire
time.

## Why, most likely

`tactics`, `defend` and `mate` all probe the **raw policy**, which STATUS already records
as a different thing from what the engine plays. `--considered 32` makes that gap wider
rather than narrower: at `--sims 32` the whole budget is spent in phase one, so sequential
halving never runs and every training target is built from 32 contenders with **one visit
each** (`mcts.rs:463-467`). That target is flatter and noisier than a halving search's, so
the raw policy sharpens less — while the engine, which always searches, is unaffected.

The network is being trained to be a good *input to search*, not a good standalone move
picker. The suites measure the latter. Widening the root made it optimise harder for the
former.

## Consequence

Three of the six metrics printed every generation — `mate`, `tactics`, `defend` — are now
known to move *against* strength at `--considered 32`. They are not merely noisy; they are
anti-correlated. Nothing in the training loop should be gated on them at this width, and
no run should be stopped because they fall.

`material` and `vs_random` moved in the right direction but far too weakly to be useful:
+7% and +1% for a change worth 137 Elo.
