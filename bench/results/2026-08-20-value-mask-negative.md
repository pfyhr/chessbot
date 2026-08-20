# Raising the ply cap: right diagnosis, worse player

**Date:** 2026-08-20 · **Machine:** Apple M2 Max

A negative result, recorded because it cost two hours and the reasoning that led
to it looked sound.

## What changed

v1 stopped self-play games at 200 plies and recorded them as draws. Measured with
a trained network, that meant **48% of games truncated and 55% of training
positions carried a result that never happened**. v2 raised the cap to 400 and
masked the value loss on whatever still truncated.

An earlier proposal — adjudicating cut-off games by material — was dropped before
it was built. Piece values are a human heuristic, and a network trained toward
them learns to hoard material rather than to win, which is the opposite of the
premise. Material is kept as a *measurement* only; the network never sees it.

## The fix did what it was for

Value head against true material, gen 24 of each run:

| material | v1 | v2 |
|---:|---:|---:|
| -9 | -0.012 | **-0.172** |
| -3 | +0.054 | -0.126 |
| +3 | +0.006 | -0.037 |
| +9 | +0.015 | **+0.140** |
| *correlation* | *+0.027* | *+0.153* |

v1's value head has no relationship with the position at all. v2's has a real if
weak one, and its endpoints separate. On its own terms the change worked.

## But the player got weaker

200 games, randomised openings, search @32:

| | result | score | Elo |
|---|---|---:|---:|
| v1 gen24 vs its own gen0 | 136W 16L 48D | 0.800 | **+241** |
| v2 gen24 vs its own gen0 | 66W 18L 116D | 0.620 | **+85** |
| **v2 gen24 vs v1 gen24** | **37W 77L 86D** | **0.400** | **-70** |

v2 improved less over its own starting point, and loses head to head.

## Why, honestly

**The comparison is confounded by compute.** Equal generations were not equal work:

| | games | positions | evaluations | wall clock |
|---|---:|---:|---:|---:|
| v1 | 6,400 | 1,089,636 | 35.6M | 61.7 min |
| v2 | 6,400 | 1,672,259 | 53.8M | **101.5 min** |

Longer games cost 64% more time for the same game count. Given v2's 101 minutes,
v1's configuration would have run roughly 41 generations instead of 25. So this is
not a clean test of the change; it is a test of the change *plus* a 40% cut in
generations.

**And the extra positions are probably poor.** The additional ~90 plies per game
come from a weak network shuffling in dead endgames. They are now about a third of
the replay buffer. Trading fabricated value labels for a large mass of
low-information positions is not obviously a good trade.

## The thing that was not fixed

Both runs show the opening metric unstable, and v2 worse: v1 oscillated between
0.111 and 0.730, v2 collapsed to **0.024** against a 0.200 random baseline — the
network actively avoids the sound first moves. That was flagged after v1 as
pointing at the training setup rather than at slow learning, and then not acted
on. A four-generation replay window with lr 1e-3 lets the network chase whatever
it saw last.

Chasing the value target while training dynamics are unstable was fighting the
wrong fire. Stability first, then re-test the cap at equal wall clock.
