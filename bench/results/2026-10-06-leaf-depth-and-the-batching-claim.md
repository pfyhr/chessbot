# The batching claim, measured: 74% at best, 45% at our default

Date: 2026-10-06. Binary sentinel `hotprof-2026-10-06-a`, M2, `drivercost
--concurrency 256 --games 256 --plies 400`.

## The claim

While reading the descent profile I wrote down an idea and labelled it
unverified:

> at `sims == considered` all 32 simulations are predetermined and could share
> one network call.

The reasoning: Gumbel's first sequential-halving round visits each considered
action exactly once, and those visits are fixed before any result returns, so
they do not need to be issued one at a time. The arithmetic worked and
`n_select_interior` at 0.25/eval was consistent with it. It was still a
derivation.

## The measurement

Counters at the one site in `mcts.rs` that returns `Status::NeedsEval` from
`create_child`, bucketing each requested evaluation by the depth of the
simulation that asked for it. Depth 1 means a direct child of the root, chosen
before any result came back from this move's search. They account for every
evaluation exactly: `root + depth1 + deeper == n_evals` in all three runs, so
nothing is hiding outside the buckets.

| considered | evals/move | depth-1/move | deeper/move | depth-1 share | mean depth | mean depth given deeper |
|---|---|---|---|---|---|---|
| 32 | 32.33 | 23.27 |  8.06 | **74.3%** | 1.260 | 2.012 |
| 16 | 32.25 | 14.10 | 17.15 | **45.1%** | 1.554 | 2.009 |
|  8 | 27.18 |  7.53 | 18.65 | **28.7%** | 1.739 | 2.037 |

## What it says

**The claim is false as written, and the reason is not search structure — it is
legality.** At `sims == considered == 32` the first round is predetermined, as
derived; there just are not 32 actions to visit. The depth-1 count per move is
`min(considered, legal moves)`, and over these games that lands at 23.27. The
shortfall grows with `considered` (0.5 at 8, 1.9 at 16, 8.7 at 32) exactly as
legality-capping predicts and as nothing else would. The remaining ~8
simulations go into a second round, which reads the first round's results.

**At our actual default (`considered = 16`) the ceiling is 45%, not 100%.**
That is the number that matters for planning, and it is half of what the
derivation implied.

**The tree is two plies deep.** Every non-depth-1 leaf is essentially exactly
depth 2 (mean 2.01-2.04 across all three settings). At 32 simulations the
search does not build a tree so much as a fan with one fold in it.

## What it is worth

The driver currently takes one leaf per game per batch, so batch width equals
concurrency. Emitting a move's whole first round at once would give 14-23 leaves
per game instead of 1: batch 3600-5900 at concurrency 256, rather than needing
thousands of concurrent games to fill a GPU. That is the lever, and the 45%
figure is the share of evaluations it can reach at our default.

What is still derived, not measured: that the second and later rounds are
batchable across distinct actions within a round. Plausible on the same
argument — a descent below one root child reads only that child's statistics —
but it is the same kind of claim that just came back 55% short, so it does not
get used as a premise.

## Note to self

The derivation was not sloppy; it was correct about the mechanism it modelled
and silent about a constraint outside it. A ratio that is exactly right about
search can still be wrong about chess. Measure it anyway — it cost one counter
and ten minutes.

---

## Correction, same evening: the depth claim above was the benchmark's, not the engine's

`drivercost` feeds **all-zero logits and all-zero values** (`drivercost.rs:45`),
deliberately — its stated job is to time the driver "with the network excluded".
Under a flat policy the interior rule `argmax[pi'(a) - N(a)/(1+sum N)]`
degenerates to round-robin: every child of a node is opened before any is
re-entered. That is the shallowest the search can possibly be. Policy sharpness
is the thing that buys depth, and this benchmark sets it to zero.

Re-measured with gen695 actually driving the search (`bench/leaf_depth.py`,
128 games, `considered = 16`), as % of leaves:

| | mean | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9+ |
|---|---|---|---|---|---|---|---|---|---|---|
| sims 32, flat | 1.55 | 44.9 | 55.4 | 0.6 | 0.0 | | | | | |
| **sims 32, real** | **1.89** | 44.9 | 36.8 | 10.3 | 5.1 | 1.2 | 0.7 | 0.5 | 0.3 | 0.4 |
| sims 128, flat | 2.08 | 10.9 | 68.7 | 19.5 | 0.1 | | | | | |
| **sims 128, real** | **3.96** | 11.5 | 26.0 | 18.3 | 13.0 | 9.6 | 6.7 | 4.5 | 3.1 | 7.2 |

**"The tree is two plies deep" was wrong.** At the budget the engine actually
plays — `sims = 128`, the UCI default — mean leaf depth is **3.96** and 7.2% of
leaves sit at depth 9 or beyond. The flat evaluator understated it by about 2×,
and understated the tail by two orders of magnitude.

**What is unaffected: everything about batching.** The depth-1 share is
`min(considered, legal moves)` divided by the simulation count — fixed by the
sequential-halving schedule, which does not read the evaluator at all. The two
measurements agree on it to within noise (44.9 vs 44.9, 11.5 vs 10.9), which is
the prediction that said they would. The 45.1% ceiling stands.

**The rule this breaks.** [[benchmark-the-loop-not-the-kernel]] says a
microbenchmark overstates speed. This is the neighbouring failure: a benchmark
built to exclude a component is valid only for questions that component does not
answer. It was the right tool for `ns_prepare` and the wrong tool for depth, and
nothing in its output said which.
