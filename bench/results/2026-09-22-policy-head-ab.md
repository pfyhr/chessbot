# The flat policy head is not waste: conv loses by 87 Elo

**Date:** 2026-09-22 · Two arms from scratch, 12.0h wall clock each, launched
together on one GPU so contention is matched. Identical in every respect except
the policy head. Booked, `openings-4ply.epd`, tc=16+0.3.

## Hypothesis, and it was wrong

`policy_fc` is a `Linear(2048, 4672)` holding **87% of the network's parameters**
while performing no spatial reasoning. The argument for replacing it with
AlphaZero's convolutional head was that those parameters are waste, and that a
conv head learns ~64x more efficiently per parameter because its filters are
shared across all 64 squares.

Predicted: conv wins, or at worst ties at 7.3x fewer parameters.

## Result

| | result | score | Elo |
|---|---|---:|---:|
| **conv-g180 vs fc-g172** | 54W 103L 43D | 0.378 | **-87 +/-42** |

Independent cross-check against a fixed, strength-matched old network:

| | score | Elo |
|---|---:|---:|
| conv-g180 vs pw32-g145 | 47.0% | -21 +/-67 |
| fc-g172 vs pw32-g145 | 64.0% | **+100 +/-68** |

The cross-checks differ by 121 Elo in favour of fc, independently consistent with
the direct -87. Two booked measurements agreeing in direction is this project's
strongest standard of evidence, and both say the same thing: **the fully
connected head is better, and the 9.57M parameters were buying something.**

## Why, most likely

`policy_loss` at the end: **conv 2.573, fc 2.268**. The conv head fits the
improved-policy target substantially worse. That is not a strength proxy, it is
a statement that the architecture cannot represent the target as well.

The likely mechanism is reach, not size. The conv head is **spatially local**:
the logits for moves leaving square X are computed from a 3x3 neighbourhood of X.
The flat head is **fully global**: every one of the 4,672 logits is a learned
function of all 64 squares at once.

Whether a move from a1 is good depends on the whole board. A trunk deep enough to
have already mixed global context into every square makes the conv head
sufficient -- which is why AlphaZero can use one on 19x256. At **6x96** the trunk
evidently does not, and the flat head's global connectivity was carrying that
load. Removing it did not free wasted capacity; it removed the only global mixing
in the network.

**Testable prediction:** the conv head should overtake as the trunk grows. The
crossover point is the interesting number, and it is unmeasured.

## The proxies lied again, sixth time

| | conv | fc | agrees with match? |
|---|---:|---:|---|
| material | **35.82** | 23.40 | **no** |
| vs-random | 0.692 | **0.733** | yes |
| tactics | 0.147 | **0.223** | yes |
| defend | 0.285 | **0.410** | yes |
| policy loss | 2.573 | **2.268** | yes |

At the 2.6h mark, material (33.4 vs 14.6) and vs-random (0.700 vs 0.608) *both*
favoured conv and were cited as an encouraging early signal. Material stayed wrong
for the whole 12 hours, ending 53% higher on the arm that lost by 87 Elo.

Material is the single most persistently misleading number in this project. It is
also the one with the most intuitive appeal, which is presumably why it keeps
getting believed.
