# Both candidates lose: the flat head earns its parameters

**Date:** 2026-09-25 · Five engines, 1000 games, 400 per engine, booked
`openings-4ply.epd`, tc=16+0.3, 4 timeouts (0.4%). C and D trained from scratch
**generation-matched at 180**, launched together so contention is matched.

| rank | engine | Elo | vs A |
|---:|---|---:|---:|
| 1 | **A-flat** (6x96 res + flat head, 10.99M) | **+112 +/-32** | -- |
| 2 | pw32-g145 (10h anchor) | +39 +/-31 | -73 |
| 3 | B-conv (conv head, 1.50M) | -13 +/-32 | -125 |
| 4 | **C-narrow** (8-channel bottleneck, 3.81M) | **-21 +/-33** | **-133** |
| 5 | **D-attn** (6 attention blocks + conv head, 1.14M) | **-118 +/-33** | **-230** |

## The hypothesis, and its falsification condition

Stated before the run: `policy_fc` holds 87% of the network's parameters while
doing no spatial reasoning, and is the *only* layer where two distant squares
meet (convolutions relay locally; `SEBlock` pools the board flat and broadcasts
one modulation everywhere). The claim was that its value is **reach, not
capacity**.

The pre-registered test: *"If C ties with A, capacity was not the point and reach
was. If C loses, the diagnosis is wrong and D's case collapses with it."*

## C lost by 133 Elo. The diagnosis is wrong.

Capacity in the policy head is doing real work. Narrowing the bottleneck from 32
channels to 8 -- which preserves full global reach and cuts only capacity --
costs 133 Elo.

**D is an independent second refutation.** It put global mixing into all six
trunk blocks, which under the hypothesis should have made the small conv head
sufficient. It finished **last**, 105 Elo below even B, whose trunk has no global
mixing at all. Adding reach to the trunk did not help the conv head; it hurt.

Two experiments, opposite directions, same verdict: the flat head's 9.57M
parameters are buying capacity, and reach is not the mechanism.

## What survives

- The **measurement** that started this: B loses to A. Confirmed twice now
  (-87 +/-42 on 2026-09-22, -125 here).
- The observation that `policy_fc` is 87% of parameters and `SEBlock` cannot
  relate two squares. Both remain true. They simply do not explain B.
- `A-flat` is ahead of the `pw32-g145` anchor by 73 Elo, so the pool is sane.

## What does not

Everything built on "global mixing" as the explanation, including section 10 of
the build log as first written, and the case for a transformer trunk *at this
scale*. Attention may still pay on a larger trunk or different hardware; nothing
here tests that. What is tested is that swapping reach in for capacity, at
6x96 and 180 generations, loses.
