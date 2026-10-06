# Capacity, three sizes: what it buys, what it costs, and one arm that cannot be used

Date: 2026-10-06, writing up runs from 3-5 Oct. Elo is fixed 128 nodes, booked
openings, 300 games per pairing (±38 Elo).

| arm | trunk | params | generations | wall clock |
|---|---|---|---|---|
| `cap-small` | 6×96 | 10.99M | 575 | 32.0h |
| `cap-big` | 8×128 | 12.45M | 386 | 32.0h |
| `cap-xl` | 10×160 | 14.83M | 100 | 9.7h |

## What capacity buys (small vs big only)

Both arms kept every checkpoint, so the 32h A/B replays as a curve.

**At equal data the bigger network wins outright, at every point:**

| matched generation | 100 | 200 | 300 | 385 |
|---|---|---|---|---|
| Elo, big minus small | +80 ±41 | +173 ±37 | +124 ±38 | +109 ±39 |

**At equal wall clock it does not:**

| share of the 32h | 12.5% | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|
| Elo, big minus small | −3 ±36 | +2 ±37 | +64 ±36 | +31 ±37 | −21 ±38 |

Two things I predicted and the data refused. The equal-time curve peaks halfway
and **ends below zero** — "give the big net more hours and it passes" is refuted
by its own data. And the equal-data advantage *fades*, 173 → 124 → 109, the
reverse of the usual picture. The fade is 1.2σ and may be noise; the positive
sign is not, being four independent points far from zero.

## What capacity costs

Median steady-state seconds per generation (first 10 generations dropped — the
window is still filling and games are longer).

| arm | trunk FLOPs | wall clock per generation | evals/sec |
|---|---|---|---|
| `cap-small` | 1.00× | 1.00× (198.0s) | 5,478 |
| `cap-big` | 2.37× | **1.49×** (294.2s) | 3,641 |

A 2.37× trunk costs 1.49× the wall clock. Capacity is substantially cheaper in
hours than in arithmetic, because a generation is not only trunk FLOPs — the
driver's CPU work and the per-call overheads do not grow with the network.
This is the quantity the equal-wall-clock curve is actually trading against,
and it had not been written down.

## The xl arm cannot be used for either

`cap-xl` has no head-to-head at all, and its cost ratio is confounded:

- `cap-small` and `cap-big` ran **concurrently** on the same laptop, 3 Oct 07:53
  → 4 Oct 15:52, contending for one machine.
- `cap-xl` ran **alone** the next day, 5 Oct 11:51 → 21:27.

Its 1.62× cost against small's 1.00× is therefore measuring "had the machine to
itself" as much as "is a bigger network", and the two cannot be separated after
the fact. The number looks like the most interesting result here — 4.63× the
FLOPs for 1.62× the time — which is exactly why it does not get reported as one.
It needs a rerun under the same contention as the arms it is compared to.

Note also that even the small/big ratio is measured under two-way contention, so
it describes *this* A/B, not what either arm would cost running alone.

## Standing

- Capacity and data pay about equally at the margin on this machine, so added
  compute should buy both in proportion. That conclusion rests on small vs big
  and is unaffected by the xl problem.
- The 10×160 question is open. Answering it costs one rerun scheduled against a
  matched-contention control, not a reanalysis.
