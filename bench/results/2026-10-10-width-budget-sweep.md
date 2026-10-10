# Root width against node budget: the optimum walks right, then hits the wall

Date: 2026-10-10. Rented RTX 3080 Ti (Vast m:33122), 37h, `bench/depth_study.sh`
plus `bench/width_budget_sweep.sh`. Every match is **gen695 against itself**,
booked openings, fixed nodes, both sides on the same budget — only
`RootActions` differs. Zero timeouts across all 1,800 games. Raw:
`data/2026-10-10-width-budget.log`.

## The grid

Elo against the `ra32` default. Positive means the narrower side won.

| nodes | ra4 | ra8 | ra16 | ra32 | mean depth at ra32 |
|---|---|---|---|---|---|
| **128** | −29 ±36 | **+41 ±25** | +26 ±40 | 0 | 3.76 |
| **512** | −115 ±44 | −57 ±35 | +21 ±52 | 0 | 6.55 |
| **1024** | −153 ±53 | −52 ±51 | −51 ±47 | 0 | — |
| **2048** | −151 ±72 | −31 ±63 | −56 ±67 | 0 | — |

The 128 row pools 600 games for `ra8` (300 on 6-7 Oct, 300 replicated 7-8 Oct);
every other cell is 200 games at 512/1024 and 100 at 2048.

## What it says

**One cell in sixteen is significantly positive.** `ra8` at 128 nodes, +41 ±25.
Everything at 1024 and above is negative, and `ra4` is catastrophic everywhere,
getting worse as the budget grows: −29 → −115 → −153 → −151.

**The optimum moves right with budget and then saturates.** At 128 nodes the best
width is 8. By 512 it has moved to 16–32 (`ra16` at +21 ±52 is a tie with the
default). From 1024 up, every narrowing loses and `ra32` is best — and there is
nowhere further to go, because the mean legal move count is **24.2**, so `ra32`
already covers 96% of the move list. The optimum has hit the ceiling of the
game, not of the search.

**The mechanism is the one the depth table predicted.** Narrowing buys depth;
depth is only worth buying while it is scarce. At 128 nodes `ra32` reaches 3.76
plies and trading width for depth pays. At 512 it already reaches 6.55 and the
trade is roughly neutral. Past that, narrowing buys depth nobody needs and pays
for it in move coverage, which is why the losses deepen rather than flatten.

## The decision

**Keep `ra32` as the default.** `ra8`'s +41 is real but it is a low-budget
special case, not a better setting. A global constant would be wrong at every
budget above 128, and we routinely use 32, 128, 256, 512, 1024 and 2048.

Worth recording for the one place it applies: **if we ever play at a fixed low
node count** — a tournament with a node cap, or a fast screening ladder — `ra8`
is worth about 41 Elo there and costs nothing to set.

Not adopting it also keeps the instrument stable: every historical Elo number in
this project was measured at `ra32`.

## What this retires

The 7 Oct write-up proposed `ra8` for match play on the strength of a single
+58 ±34 cell. That has now been replicated down to +41 pooled, shown to reverse
at 512, and shown to lose at every budget from 1024 up. Three successive
measurements each made the claim smaller and narrower. See
[[best-of-n-needs-replication]].

The self-play side of that document is untouched: `ra32` there rests on move
coverage — 96% of legal moves evaluated per turn at 32 simulations — which this
sweep does not bear on.
