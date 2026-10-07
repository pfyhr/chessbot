# What search is worth, and the root width that wins for free

Date: 2026-10-07, run overnight on aida gpu0 (borrowed RTX 3090).
`bench/depth_study.sh`. Every match is **gen695 against itself**, booked
openings, fixed nodes, so these measure search alone with the network held
constant. Raw log: `runs/night-depth/report.log` on aida.

## 1. Leaf depth, with the real network driving

Mean leaf depth. Full histograms in `data/2026-10-07-leaf-depth.json`.

| sims | ra8 | ra16 | ra32 |
|---|---|---|---|
| 32 | 2.58 | 1.87 | **1.57** |
| 64 | 3.46 | 3.13 | 2.38 |
| 128 | 4.68 | 4.14 | **3.76** |
| 256 | 6.26 | 5.28 | 5.12 |
| 512 | 8.44 | 7.36 | 6.55 |

Depth grows about 1.3 ply per doubling of simulations, and narrower roots are
deeper at every budget. The bolded cells are what we actually run: self-play at
32/ra32, match play at 128/ra32.

**Why width costs depth.** Traced from `begin_phase` and `next_root_edge` at
`sims = 32`, where `visits_each = max(sims / (phases * m), 1)`:

| | phase 1 | phase 2 | phase 3 | sims used | max visits to one root child |
|---|---|---|---|---|---|
| `ra32` | 32×1 | — | — | 32 | **1** |
| `ra16` | 16×1 | 8×1 | 4×2 | 32 | **4** |
| `ra8` | 8×1 | 4×2 | 2×5 | 26 | **8** |

Every contender costs one simulation before anything is deepened — a flat tax of
`m` off the top. At `ra32` that tax *is* the whole budget, so no node below the
root is ever created; 70.2% of evaluations are depth-1 first visits. Depth then
responds to visits *per node*, not total visits: `select_interior` re-enters a
known child only once `pi'(best) - N/(1+sum N)` beats the best unopened sibling,
needing roughly `1/pi'_best` visits per ply, so depth `d` costs about
`(1/pi'_best)^d` visits concentrated on one line.

The schedule predicts the measured depth-1 shares: 72.8% vs 70.2% at `ra32`
(using the 23.3 mean legal moves), 44% vs 44.6% at `ra16`, 31% vs 28.6% at `ra8`.

## 2. What a doubling of search is worth

Positive means the bigger budget won.

| rung | Elo | games |
|---|---|---|
| 64 over 32 | **+248 ±43** | 300 |
| 128 over 64 | +100 ±39 | 300 |
| 256 over 128 | +90 ±44 | 240 |
| 512 over 256 | +59 ±50 | 160 |
| 1024 over 512 | +42 ±65 | 100 |

**+539 Elo cumulative** from 32 to 1024 nodes on one unchanged network.
Diminishing but positive throughout, except the last rung, which does not clear
its own error bar. Cross-check: the first two rungs sum to +348 against +338
from an independent 128-vs-32 calibration duel.

## 3. Breadth versus depth at a fixed 128 nodes

Same budget on both sides, only `RootActions` differs. Positive means the
narrower, deeper side won.

| | Elo | games |
|---|---|---|
| ra4 over ra32 | −29 ±36 | 300 |
| **ra8 over ra32** | **+58 ±34** | 300 |
| ra16 over ra32 | +26 ±40 | 300 |

**An interior optimum around 8.** Narrowing from the default 32 to 8 is worth
about 58 Elo at the same node budget; narrowing further to 4 gives it back. Only
`ra8` clears its own error bar, but the inverted-U across all three is more
persuasive than any single cell, and it matches the depth table — `ra8` at 128
sims searches to 4.68 plies against `ra32`'s 3.76.

**This is free.** `RootActions` is a UCI option, not a training choice. Three
cautions before adopting it:

- Three comparisons were run, so one false positive at 95% is roughly a 14%
  risk. The shape argues against that being what happened here; a confirmatory
  match would settle it.
- Measured at one budget (128 nodes) on one network (gen695). It may not hold
  elsewhere.
- Every past Elo number in this project was measured with the engine at `ra32`.
  Changing the default changes the instrument, so old and new results stop being
  comparable unless the change is dated and recorded.

## What this does *not* answer

**It does not say what self-play simulations should be.** Self-play search does
two jobs — choose the move, and produce the improved-policy training target —
and every number here measures only the first, on a network that is already
trained. The two come apart, and we have direct evidence of it: root actions
16→32 is in the log at **+160 Elo of trained strength**, and the depth table
shows that same change makes the tree *shallower* (1.87 → 1.57). If training
wanted depth, that should have hurt.

There is a reason it might not. At `sims = 32`, `ra32` sequential halving never
halves: every root move gets exactly one visit, so the improved policy is a clean
one-ply policy improvement over all legal moves — the regime Gumbel MCTS was
built for. "70.2% of evaluations never leave depth 1" reads like a defect and may
be the design working.

Answering it needs the instrument that worked for capacity: a wall-clock-matched
training A/B, `sims 32` against `sims 64`, both arms sharing one GPU under
identical contention, then a booked head-to-head plus the equal-data and
equal-wall-clock curves. See [[ab-arms-must-share-the-machine]].

## Loose end

`next_root_edge` returns `None` once `contenders.len() <= 1`, so halving down to
a single survivor ends the search with budget unspent. The traced schedule says
that is 6 of 32 simulations at `ra8` — 19% — and about 5% at 128. That is a
derivation, not a measurement, and gets a counter before it gets a number.
