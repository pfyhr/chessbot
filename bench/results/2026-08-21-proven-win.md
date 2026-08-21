# Proven-win propagation: the value head was drowning out checkmate

**Date:** 2026-08-21

## The symptom

After ten hours of training, the engine found a mate in one **26%** of the time. With a
*flat* network — uniform policy, zero value — the same search found it **97.5%** of the
time. Training had made the engine worse at forced mates than knowing nothing.

## The mechanism

A terminal child's exact value was backed up like any other number and then passed
through `sigma_completed`, which min-max rescales Q across a node's children. Concretely,
with a certain mate, one confidently-wrong network estimate, and a bad move:

| edge | Q | rescaled | x scale 5.1 |
|---|---:|---:|---:|
| checkmate | +1.00 (certain) | 1.000 | 5.10 |
| move the net likes, wrongly | +0.95 (noise) | 0.974 | 4.97 |
| worst move | −0.90 | 0.000 | 0.00 |

Certain checkmate beat confident noise by **0.13** — against Gumbel noise of σ ≈ 1.28 and
policy logit gaps of 2–5. The mate was routinely outvoted.

The flat network scored well precisely *because* it had no opinions: the mate's +1 was
the only non-zero value, so rescaling gave it 1.0 against 0.0 everywhere. Training fills
that range with competing confident numbers and compresses the certainty away. Given the
value head's measured correlation with ground truth (+0.027), those numbers are noise.

## The change

MCTS-Solver (Winands et al. 2008). `Edge::proven` and `Node::proven` record what the
rules guarantee; `Search::solve` combines them — one winning move proves a win, while
proving a loss needs every move settled and losing. Three consumers: root ranking (a
proven result is offset outside the reachable score range), the improved policy (a proven
win becomes the target outright), and interior selection (settled moves are skipped).
Settled nodes also short-circuit, so the budget stops draining into solved lines.

This adds no chess knowledge. "This position is checkmate" is the definition of the game,
not a heuristic like piece values — `expand()` already computed it. The change stops the
search averaging that fact with a guess.

## Result

Mate-in-1 accuracy, 120 positions, ~40 legal moves each, random baseline 0.034:

| setting | before | after |
|---|---:|---:|
| trained net, 32 sims / 16 root actions | 0.158 | 0.367 |
| trained net, 64 sims / 16 root actions | 0.217 | 0.367 |
| trained net, 32 sims / 64 root actions | 0.233 | 0.675 |
| **trained net, 64 sims / 64 root actions** | **0.217** | **1.000** |
| trained net, 128 sims / 64 root actions | 0.258 | 0.992 |
| flat net, 128 sims / 64 root actions | 0.975 | 1.000 |

The trained network now matches the flat one.

## A second finding, not yet acted on

`max_considered` matters more than expected. At 16 root actions with ~40 legal moves the
mating move is often never sampled at all, capping accuracy at 0.367 however many
simulations are spent. At 64 it reaches 1.000. Raising the default is not free — Gumbel's
premise is considering few actions well, and sequential halving spreads a fixed budget
thinner across more of them — so it needs a training comparison rather than a unilateral
change.

## On the Gumbel guarantee

The paper's policy-improvement guarantee assumes π′ is derived from completed-Q.
Overriding it with proven results steps outside that proof. A proven win cannot be a
degradation — there is nothing better than winning — but the departure is real and worth
stating.

## Proving a loss is harder than proving a win

A winning move is itself terminal, so one visit settles it. A losing move is not: the
search must descend into it, expand it, and reach the opponent's winning reply — for
*every* alternative — before the position is solved. Measured on a Connect4 must-block
position over 100 seeds: 16 sims blocks 18% of the time, 32 sims 40%, 64 sims 62%, 128
sims 100%.
