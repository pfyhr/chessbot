# Proven-win propagation and search width: two 10-hour arms

**Date:** 2026-08-22 · **Machine:** Apple M2 Max
Both arms run concurrently, as the baseline did, so contention is matched.

All three runs share a configuration: 200-ply cap, no value masking, window 8, lr 5e-4,
32 simulations per move. They differ in two things only.

| run | proven-win | root actions | generations | games | evaluations |
|---|---|---:|---:|---:|---:|
| baseline (`wc-capped`) | no | 16 | 145 | 37,120 | 199M |
| `pw-16` | **yes** | 16 | 136 | 34,816 | 175M |
| `pw-32` | **yes** | **32** | 146 | 37,376 | 161M |

## Result

| | baseline | pw-16 | pw-32 |
|---|---:|---:|---:|
| material vs random | +14.7 | +14.4 | **+21.6** |
| defend (base 0.229) | 0.375 | 0.380 | **0.405** |
| truncated games | 36% | 19% | **14%** |
| mean game length | 165 plies | 146 | **123** |
| value/material correlation | +0.282 | +0.281 | +0.232 |

Head to head, 200 games, randomised openings, 32 sims / 16 root actions for both sides:

| | result | score | Elo |
|---|---|---:|---:|
| pw-16 vs baseline | 100W 72L 28D | 0.570 | **+49** |
| pw-32 vs baseline | 128W 42L 30D | 0.715 | **+160** |
| pw-32 vs pw-16 | 110W 53L 37D | 0.642 | **+102** |

## Reading it

**Proven-win propagation helps on its own, modestly.** pw-16 differs from the baseline in
exactly one thing and gains +49 Elo. Its clearest effect is conversion: truncation falls
36% → 19% and games shorten by 20 plies. The engine finishes what it starts.

The three margins are mutually consistent: +160 against the baseline, +49 for proven-win
alone, and +102 head to head between them.

**Search width is the larger lever.** pw-32 nearly doubles the baseline's material and
gains +160 Elo, with truncation down to 14% and games 40 plies shorter. It does this on
*fewer* evaluations (161M vs 199M) — considering more root actions costs simulations per
action, and it still comes out well ahead.

The mechanism was visible in the mate-in-1 suite before either run started: at 16 root
actions with ~40 legal moves the winning move is frequently never sampled, so accuracy
caps at 0.367 no matter the budget; at 64 it reaches 1.000. That is not specific to mates.
Any position with one good move and many bad ones has the same problem, and conversion is
full of them.

There is a tension worth naming. Gumbel's premise is that considering few actions *well*
beats considering many badly, and the paper's sample-efficiency argument rests on it. That
argument assumes the shortlist is chosen by a policy worth trusting. At +0.28 correlation
between value and ground truth, the shortlist is close to arbitrary, and widening it is
the cheaper repair. A better network might well flip this back.

## The value head is no longer the problem it was

Correlation with true material is +0.28 across all three runs, against **+0.027** for the
earlier 25-generation run. That improvement came from training length, not from any of the
changes tested here — and it happened even in the arm that trains on ~36% invented labels.

Note that pw-32 has the *lowest* correlation (+0.232) and is comfortably the strongest
player. Material is a measurement, not the objective; a network that ranks positions by
something other than material can be better at winning, which is the entire premise.

## Still unexplained

Opening preference collapsed again: pw-16 ends at 0.387, pw-32 at 0.054, against a 0.200
random baseline. Four hypotheses have now failed — replay window, learning rate, value
target quality, and search width. There is no working theory for this.

Mate-in-1 on the training-time suite also stays near baseline (0.035 and 0.070 against
0.031), even though proven-win takes the *engine* to 1.000 at 64 root actions. The suite
measures the raw policy, which is a different thing from what the engine plays.
