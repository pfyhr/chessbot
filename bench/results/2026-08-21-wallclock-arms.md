# Ten hours a side: does the ply cap cost anything?

**Date:** 2026-08-21 · **Machine:** Apple M2 Max
Both arms run concurrently on one GPU so they contend equally.

| | A — cap 200, draw recorded | B — cap 400, label dropped |
|---|---:|---:|
| generations | 145 | 89 |
| games | 37,120 | 22,784 |
| evaluations | 199M | 190M |
| material vs random | −1.8 → **+14.7** | −2.5 → +9.1 |
| vs random | 0.467 → **0.708** | 0.467 → 0.542 |
| defend (base 0.229) | 0.235 → **0.375** | 0.200 → 0.340 |
| truncated games | 43% → 36% | 17% → **3%** |
| Elo over own gen 0 | **+531** | +241 |

Evaluations match within 5%, so this is a fair compute comparison — unlike the earlier
attempt, which gave both arms 25 generations and silently handed A a 40% discount
because its shorter games made a generation cheaper.

## More training matters, a lot

Both arms far exceed their one-hour predecessors: A's self-improvement went +241 → **+531
Elo**, B's +85 → **+241**. Nothing in the earlier one-hour runs was near convergence, so
the conclusions drawn from them were premature.

## The prediction that failed

The claim was that A, training on ~36% invented value labels, would climb faster early and
then stall, while B's clean signal kept paying. Measured over the final three hours:

| | material gained per hour |
|---|---:|
| A | +0.85 |
| B | +0.89 |

**Neither is flattening, and their recent rates are equal to within noise.** A leads by an
offset it built early, not by improving faster now. The stall has not appeared, and on this
evidence there is no reason to expect it.

What B does buy is games that finish: truncation falls 17% → 3% with the cap held fixed,
which is the network learning to convert rather than shuffle. A improves far less and stays
above a third, because its tighter cap keeps cutting games short whatever the network does.
Whether that eventually matters is still unmeasured — an Elo-versus-wall-clock curve for
both arms against a fixed anchor is the measurement that would settle it, and it is slow
enough that it is still running.

## The unresolved problem

Opening preference swings rather than converges, in both arms. A peaks at 0.575 around
five hours and ends near **0.024** — below the 0.200 random baseline, actively avoiding the
moves it had learned to prefer. B oscillates between 0.045 and 0.763.

This was blamed on the replay window and the learning rate. Both were changed for these
runs — window 4 → 8, lr 1e-3 → 5e-4 — and it happened anyway. That diagnosis was wrong and
the cause is unknown.

## Also still unresolved

Mate-in-1 stays at its 0.031 random baseline in both arms after 37,000 games. The suite is
drawn from random walks, so it is a hard out-of-distribution probe; at this scale it may
simply be unreachable, in which case it is not earning its place as a gate.
