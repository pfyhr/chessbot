# Architecture rankings at 20 generations invert by 200

**Date:** 2026-09-24 · Four architectures from scratch, generation-matched at 20,
128 games/gen, ~1.9h each under four-way contention. Booked round robin,
120 games per engine, `tc=8+0.15`, zero timeouts, zero crashes.

## The control failed, which is the result

`B` (convolutional policy head) was included as a **positive control, not a
candidate**. Its answer is already known from a 12h matched A/B on 2026-09-22:
**B lost to A by 87 +/-42 Elo**. The question was whether a 2h screening run
could reproduce a difference we know exists.

| rank | engine | Elo | score |
|---:|---|---:|---:|
| 1 | **B-conv** | **+176 +/-69** | 73.3% |
| 2 | D-attn | +127 +/-63 | 67.5% |
| 3 | **A-flat** | **-92 +/-64** | 37.1% |
| 4 | C-narrow | -219 +/-74 | 22.1% |

It did not merely fail to reproduce it. **It inverted it.** B finishes ~268 Elo
*above* A here against 87 below at 12h -- a ~355 Elo swing, with non-overlapping
error bars at both ends.

## Ruled out

- **Not clock starvation.** Zero losses on time across 240 games, so the larger
  flat-head network was not being squeezed by the time control.
- **Not noise.** +/-69 and +/-64 on the two engines concerned; the gap is ~4 sigma.
- **Not a crash or a mis-loaded architecture.** Zero crashes; `infer_arch` round
  trips are unit-tested.

## What it means

**Short-run architecture screening does not work here, and is not merely noisy --
it is anti-correlated with the converged answer.** At 20 generations every
network is barely trained, and whatever ranks them at that point is not what
ranks them at 200.

A partial mechanism: B and D carry ~90k-parameter policy heads that converge
quickly, while A's flat head has 9.57M parameters nowhere near fitted after 20
generations. Early training would then reward fast-converging small heads. But
C, whose head is 2.39M -- between the two -- finishes *last*, which that story
does not explain. No clean mechanism is claimed.

## Consequence

There is no cheap version of the architecture experiment. A 2.4h screen cannot
stand in for a 12h run, and any future proposal to "quickly check" an
architecture change at low generation counts should be refused on this evidence.

The 12h matched A/B remains the minimum credible budget, and where throughput
differs between arms, they should be **generation-matched rather than
wall-clock-matched** -- because training length is now demonstrated to reorder
architectures, so unequal generations is exactly the confound this result warns
about.

Cost of learning this: 2.4 hours. Cost of not knowing it: a 12h run read the
wrong way round.
