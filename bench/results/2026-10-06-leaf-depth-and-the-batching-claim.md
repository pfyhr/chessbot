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
