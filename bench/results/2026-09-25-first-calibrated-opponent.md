# A calibrated opponent at last: Lc0 with a 16x2 net at one node

**Date:** 2026-09-25 · 40 booked games per rung, `openings-4ply.epd`,
**fixed nodes rather than a clock**, zero timeouts across 160 games.

Every rating in this project has been relative to a random mover, and that anchor
is saturated: random scores ~1% and the implied gap swings ~240 Elo on a couple of
half-points. No standard engine was weak enough to replace it -- Stockfish floors
at 1320 and wins 72/72 even at a single node, because node-limiting removes its
search but not its NNUE evaluation. Strong Lc0 nets have the same floor: T40 at
one node still plays ~2300.

A **16x2** Lc0 net (`11258-16x2-se-4`, 2 blocks x 16 filters, openly published)
has no such floor. At low node counts its evaluation *is* the tiny net.

## Sweep: `weekend/gen1371` at 128 nodes vs lc0-16x2 at N nodes

| lc0 nodes | our score | Elo | usable? |
|---:|---:|---:|---|
| **1** | **37.5%** | **-89 +/-66** | **yes -- squarely in band** |
| 4 | 21.3% | -228 +/-103 | yes, marginal |
| 16 | 7.5% | -436 +/-209 | saturating |
| 64 | 0.0% | -inf | saturated |

**At one node we score 37.5%.** That is the first opponent in this project's
history that is neither crushed nor crushing -- a real rung, with a real error bar.

## Fixed nodes, not a time control

Both engines honour `go nodes`, so each side gets an identical search budget
however loaded the machine is. This removes the contention sensitivity that
probably compressed the 2026-09-20 measurement by ~100 Elo (a contended match read
+137 +/-49 where an idle one read +241 +/-73 on the same pair). Zero timeouts
across 160 games, against 3-in-4 on the first contended attempt at tc=2+0.05.

**This should be the house protocol.** Node-limited matches are reproducible on a
busy machine; time-controlled ones are not.

## Why it matters beyond one number

Lc0's own Elo chart carries our exact defect -- "the first net is set to Elo 0, so
it is not comparable, even between different training runs" -- but its later nets
are anchored to CCRL against engines of known rating. So the chain exists:

> ours -> lc0-16x2 at 1/4/16 nodes -> larger Lc0 nets -> CCRL -> an absolute rating

Each link can be kept inside 15-85%. That is the overlapping ladder this project
has needed since the random-mover anchor was found to be saturated, and most of it
is already built and rated by someone else.

Nets are in `runs/nets/` (gitignored); `brew install lc0` supplies the engine.
