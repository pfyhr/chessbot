# More self-play search buys a better network per game, and barely one per hour

Date: 2026-10-21. Training on a rented 2×RTX 4090, **one arm per card**, 13.6h
wall clock. Matches at 128 nodes, `RootActions 32`, booked openings, run on the
same box. 15 duels, 1,900 games, zero timeouts. Raw:
`data/2026-10-21-sims-curves-13h.log`, `data/2026-10-21-sims-curves.json`,
both arms' `history.json`. Checkpoints archived off-box (4.4 GB, 35 files,
MD5-verified).

Positive Elo = **sims32**, the cheaper arm, won.

## Equal data — the quality question

Matched generations. Both arms played 256 self-play games per generation, so at
gen *N* the two networks have seen the same number of games.

| generation | Elo | games |
|---|---|---|
| 50 | −17 ±62 | 100 |
| 100 | −96 ±64 | 100 |
| 200 | −191 ±81 | 100 |
| **300** | **−402 ±123** | 100 |
| 400 | −252 ±77 | 100 |
| 500 | −164 ±74 | 100 |
| 550 | −186 ±72 | 100 |

**Doubling self-play search produces a dramatically better network per game of
data** — mean **−239 Elo** from gen200 on, peaking at −402. This is the largest
effect this project has measured from any single setting.

It also **reverses the early-training result** from the 19 Oct salvage, where
sims32 led by +101 ±37 at gen 50–100. Here gen50 is −17 ±62 and gen100 is
−96 ±64. The crossover is much earlier, or absent. One difference is visible and
plausible: that run shared one GPU and this one gave each arm a card, so the
earlier arms were competing. Whether that explains it is untested.

## Equal wall clock — the question that decides anything

| hours | pairing | Elo |
|---|---|---|
| 1.50 | s32-g100 vs s64-g50 | **+173 ±61** |
| 2.89 | s32-g200 vs s64-g100 | +21 ±52 |
| 4.90 | s32-g350 vs s64-g200 | −28 ±54 |
| 6.20 | s32-g450 vs s64-g250 | −63 ±52 |
| **7.51** | s32-g550 vs s64-g300 | **−80 ±56** |
| 9.45 | s32-g700 vs s64-g400 | −75 ±52 |
| 10.74 | s32-g800 vs s64-g450 | −61 ±54 |
| 12.66 | s32-g950 vs s64-g550 | −37 ±57 |

**The per-hour advantage is a tenth of the per-game one, and it decays.**
`sims32` wins decisively for the first ~4h, `sims64` leads by at most 80 Elo
around 7.5h, and by 12.66h the gap has shrunk to −37 and is heading back toward
zero.

The mechanism is arithmetic: `sims64` costs **1.76×** per generation (measured
sustained, 86.5s against 49.1s), so at a matched hour it has seen roughly half
the games. A −239 Elo quality advantage per game, minus the cost of half the
data, nets out to −37…−80.

**One point replicates the earlier run exactly.** At ~4.9h this gives −28 ±54;
the 19 Oct salvage gave **−28 ±50** at 4.95h — different hardware, different
training run, different opponents. That agreement is the strongest evidence we
have that the wall-clock curve measures something real.

## The decision

**Keep `sims 32` for self-play.** At every budget up to ~5h it is clearly better,
and beyond that `sims64`'s edge never exceeds 80 Elo and is shrinking by 12h.
Against that, doubling search doubles the cost of every future experiment.

The honest caveat: beyond 12.66h is unmeasured, and the curve's late direction
(back toward zero) is based on two points.

## What this retires: four extrapolations, all too large

| claim | predicted at ~10.7h | measured |
|---|---|---|
| 19 Oct log-linear fit | −267 | **−61** |
| 21 Oct 4-point fit (mine) | −161 | **−61** |

Both fits were anchored on early points that genuinely looked log-linear, and
both were wrong because the curve is not a line — it has a peak. The earlier fit
was also built on a **1.53× cost ratio that was a contention artifact**: both
arms shared one GPU, which slowed `sims32` and flattered `sims64`. The honest
ratio on dedicated cards is 1.76×.

A third miss, same day: the one-generation probe put the ratio at 1.96× because
it paid CUDA startup in full. Three measurements of the same quantity gave 1.53,
1.96 and 1.76; only the sustained one was right.

See [[derivation-is-not-measurement]]. The rule earned another instance: a trend
fitted inside a narrow range says nothing about the range outside it, and this
is now the fourth time in this project that extrapolating a clean early slope
produced a confident wrong number.

## Cost of getting here

The run was meant to be 24h and died at 13.6h to a **watchdog bug of mine**: it
counted generations with `^ +[0-9]+`, the field is right-aligned in 4 characters,
and at gen 1000 the leading space vanished so the counter froze. An `||` across
both arms then condemned the healthy one, and it killed on the counter alone
while logging `state: Rl+ / Rl+` — both running. 10.4h of training lost, plus
~11h of idle billing because nothing told me it had stopped.

Fixed: padding-proof regex, per-arm tracking, and no kill without corroboration
from `/proc/<pid>/stat` showing no CPU burned. See [[pull-checkpoints-inline]]
for the other lesson the same run taught.
