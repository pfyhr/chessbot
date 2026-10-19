# Depth at the top of the budget: the published mechanism held, the rate did not

Date: 2026-10-19, local M2 Max, `bench/leaf_depth.py`, 64 games per cell,
**gen695** driving the search at `RootActions 32`. Raw:
`data/2026-10-19-leaf-depth-high.json`.

The 10 Oct width sweep explained its own results with a depth argument that
reached past its data. Depth had been measured at 32–512 simulations; the sweep's
two most decisive rows were at **1024 and 2048**, where it had not. This closes
that gap.

## The measurement

| sims | mean leaf depth | ply per doubling | `visits_each` | d9+ share |
|---|---|---|---|---|
| 32 | 1.57 | — | 1 | — |
| 64 | 2.38 | +0.81 | 1 | — |
| 128 | 3.76 | +1.39 | 1 | — |
| 256 | 5.12 | +1.35 | 1 | — |
| 512 | 6.55 | +1.44 | 3 | — |
| **1024** | **8.95** | **+2.40** | 6 | 42.1% |
| **2048** | **11.63** | **+2.68** | 12 | 55.5% |

Mean depth comes from a dedicated sum counter, not from the histogram, so the
`d9+` bucket does not censor it.

## What it confirms

**The mechanism in the sweep write-up stands.** Narrowing the root buys depth,
and depth is only worth buying while it is scarce. At 1024 nodes `ra32` on its
own already searches to **8.95 plies**, with 42% of leaves at depth 9 or beyond.
Spending move coverage to buy more depth there is paying for something the search
already has, which is why every narrowing loses at 1024 and above.

**The prediction was directionally right and quantitatively wrong.** Before the
run I extrapolated the 1.3–1.4 ply-per-doubling trend and said 1024 would land
near 8 plies and 2048 near 10. It came in at 8.95 and 11.63 — the trend line
understates the top of the range by about 16%, because the rate is not constant.

## Why the rate is not constant

Depth per doubling sits at ~1.4 ply from 128 to 512, then jumps to **2.4–2.7**.
The jump tracks `visits_each = max(sims / (phases * m), 1)` (`mcts.rs:522`)
coming off its floor. At `ra32`, `phases = 5` and `m = 32`, so the denominator is
160: every budget up to 256 gets `visits_each = 1`, clamped by the `max`.

With one visit per root child per phase, the schedule is forced to round-robin
across contenders and depth accumulates only through the survivors of successive
halvings. Once the budget buys 6 or 12 visits each, consecutive simulations land
on the *same* root child and descend a principal variation that is already
carved — each one extends an existing line rather than widening a shallow
frontier. Depth compounds instead of accruing.

This also explains the low end, where the `max(…, 1)` clamp is doing the work:
`ra8` at 128 nodes has `visits_each = 5` while `ra32` has 1. That is the whole
advantage of `ra8` at 128 nodes, and it evaporates at 1024, where `ra32` reaches
6 on its own.

## What it changes

Nothing operational — `ra32` remains the default, for the reasons in the sweep.
What changes is one sentence of published reasoning, which rested on an
extrapolation and now rests on a measurement, and one retired habit: "depth grows
about 1.3 ply per doubling" was true over the range it was measured on and false
outside it. It is a schedule artifact, not a law.

See `2026-10-10-width-budget-sweep.md` and [[derivation-is-not-measurement]].
