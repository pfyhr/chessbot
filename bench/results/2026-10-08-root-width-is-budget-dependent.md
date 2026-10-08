# `ra8` does not generalise: root width has to scale with budget

Date: 2026-10-08, M2 overnight, `bench/confirm_root_width.sh`. Both sides
gen695, booked openings, fixed nodes, 300 games per duel, zero timeouts. This
amends `2026-10-07-split-root-width.md`, whose match-side conclusion was too
strong.

## What the confirmation was for

The 6-7 Oct study put `ra8` at +58 ±34 over the `ra32` default at 128 nodes, and
two things kept that from being adoptable. Both were tested, and **both landed
against it**.

## 1. The replication halved the effect

| at 128 nodes | Elo | games |
|---|---|---|
| 6-7 Oct, aida (best of three comparisons) | +58.5 ±34 | 300 |
| 7-8 Oct, M2 (the `ra8` cell alone, fresh opening draw) | **+23.2 ±35** | 300 |
| **pooled** | **+41.2 ±25** | 600 |

The two are the same measurement (z = 0.72), so nothing is broken — but the
replication **does not clear zero on its own**, and the pooled estimate is
**+41, not +58**.

This is the winner's curse, and it is exactly what the multiple-comparison
caveat was about. `ra8` was reported because it was the best of `ra4`, `ra8`,
`ra16`; selecting the maximum of three noisy estimates biases it upward. The
effect is real — pooled, it clears zero — it is just about **30% smaller** than
the number that nearly got adopted.

## 2. At a larger budget it reverses

| at 512 nodes | Elo | games |
|---|---|---|
| `ra8` over `ra32` | **−57.3 ±35** | 300 |

Not a shrinking margin: a significant loss. The prediction was that narrowing
only pays while depth is scarce, and the depth table says exactly when that
stops being true — `ra32` reaches **3.76** plies at 128 sims but **6.55** at 512.
Once the tree is deep enough, narrowing buys no depth worth having and the
coverage it costs dominates.

## What this means

**The optimum root width is a function of the node budget, and it grows with it.**
`ra8` is not a better default; it is the better setting *at around 128 nodes*.
There is no single number to adopt.

**Recommendation: leave the default at `ra32`.** Not because the gain is unreal —
pooled, +41 ±25 at our match budget is real — but because:

- a global default is wrong at any budget we are not currently using, and we
  used 32, 128, 256, 512 and 1024 nodes this week alone;
- the estimate has already halved once under replication;
- every historical Elo number here was measured at `ra32`, so switching costs
  comparability for a modest, budget-specific gain;
- the optimum also moves with policy sharpness
  ([[recheck-root-width-as-the-net-sharpens]]), so it would need re-deriving per
  champion *and* per budget.

**The self-play side is untouched.** `ra32` there is decided by move coverage —
96% of legal moves evaluated per turn at 32 simulations — not by depth, and
nothing here bears on it. That half of `2026-10-07-split-root-width.md` stands.

## Worth doing instead

A width-against-budget sweep: `ra4/8/16/32/64` at 128, 512 and 2048 nodes, which
would locate the ridge rather than sampling one point on it. Roughly 15 duels at
~300 games — a day on one GPU, no retraining. If the optimum tracks budget
cleanly, the right change is a *rule* (width as a function of nodes) rather than
a constant, and that is worth changing the instrument for.
