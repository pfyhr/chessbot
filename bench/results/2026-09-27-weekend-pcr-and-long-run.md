# PCR loses by 200 Elo; 32 more hours of plain training gains 91

**Date:** 2026-09-27 · Unattended three-phase chain, ~44h GPU. All matches booked
on `openings-4ply.epd` at **fixed nodes** (128 per side) rather than a clock, so
nothing here depends on how loaded the machine was. Zero timeouts throughout.

## Phase 1-2: playout cap randomization

Two arms from scratch, 12h wall-clock matched, launched together.

| arm | config | generations |
|---|---|---:|
| control | `sims 32 / considered 32` | 224 |
| PCR | `sims 64 / considered 32`, p=0.25, fast `20/20` | 209 |

Settings were chosen so the two cost the same: measured beforehand at 97.8 vs
96.1 s/gen, and `full=64` is the smallest budget at which sequential halving
actually runs at m=32 (32 -> 16 -> 8 -> 4).

| | result |
|---|---|
| **PCR vs control, head to head** | **-200 +/-50 Elo** (200 games) |
| PCR vs `lc0-16x2@1node` | 30.0% |
| control vs `lc0-16x2@1node` | 33.0% |

Two independent measurements agreeing in direction. **PCR is decisively worse at
this scale.**

### Why, most likely

PCR trained the policy head on ~3.5x fewer positions per generation (25% of
44,496 against 100% of 39,538), and completed fewer generations despite being
marginally faster per generation. With batch 512 that is ~128 positions carrying
policy gradient per step instead of 512. KataGo's bet is that better targets
outweigh fewer of them; at 6x96 they do not.

This is **not** "PCR does not work". It is "PCR at p=0.25, 6x96, batch 512 does
not work". Raising p to 0.5 would halve the data penalty. But this is the fourth
consecutive borrowed technique to fail here, after the conv head, the narrow
bottleneck and the attention trunk, and the pattern is consistent: what pays at
KataGo's and Lc0's scale does not pay at ours.

### A false alarm, retracted

`known_fraction` fell to 0.677 in the first PCR smoke test and was written up as
evidence that cheap moves were pushing games into the ply cap and destroying
value targets. Measured side by side against a control: **control 0.818, PCR
0.847.** PCR loses *fewer* value targets. The original number was an untrained
network playing long games, read without a control beside it.

## Phase 3: 32 more hours of ordinary training

Resumed `weekend/gen1371` with the control configuration for 32h, 696 generations.

| | result |
|---|---|
| **`w-long/gen695` vs `weekend/gen1371`** | **+91 +/-47 Elo** (200 games, 62.8%) |
| `w-long/gen695` vs `lc0-16x2@1node` | 34.7% (150 games) |

**No saturation.** Cumulative training went 77h -> 109h, half a doubling, for
+91 Elo. That is 88-275 Elo per doubling depending where in the error bar it
lands, against 126-167 measured earlier: the rate is holding.

### The proxies were flat the whole time

Across all 696 generations: material 32.3 -> 32.3, tactics 0.207 -> 0.195,
vs-random 0.817 -> 0.820, policy loss unmoved. Read at face value, 32 hours
bought nothing. It bought 91 Elo. **Eighth instance.**

### The anchor needs more games than it was given

`gen1371` scored 37.5% against `lc0-16x2@1node`, and `gen695` -- which beats it
by 91 Elo head to head -- scored 34.7%. Not a contradiction: the 37.5% came from
**40 games** (+/-66 Elo), far too coarse to resolve a 91-Elo step. The 150-game
reading is +/-35.

**Rule for the ladder: 150+ games per rung.** A 40-game rung cannot see the size
of step this project now makes in a weekend.
