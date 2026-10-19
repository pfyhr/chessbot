# Self-play search: the cheap arm wins early and loses late

Date: 2026-10-19, local M2 Max, `bench/sims_equal_data.sh`. Both sides play at
**128 nodes, `RootActions 32`** — identical match search. The only difference is
the search budget each network was *trained* with: **32 vs 64 simulations per
self-play move**. Raw: `data/2026-10-19-sims-equal-data.json`,
`runs/sims-equal-data/report.log`.

Positive Elo means **sims32**, the cheaper arm, won.

## The curve

| matched generation | Elo | games | score |
|---|---|---|---|
| 50 | **+136 ±56** | 150 | 68.7% |
| 100 | **+68 ±50** | 150 | 59.7% |
| 200 | **−139 ±54** | 150 | 31.0% |
| 300 | **−90 ±50** | 150 | 37.3% |
| 400 | **−113 ±53** | 150 | 34.3% |

Pooled: **+101 ±37 early** (gen 50–100, 300 games), **−114 ±30 late**
(gen 200–400, 450 games). Both clear their error bars, with opposite signs.

Matched generations are equal *data*: both arms played 256 self-play games per
generation, so at gen *N* the two networks have seen the same number of games.
Confirmed in the histories — 262,060 against 251,724 training samples at gen400,
while `evaluations` differ by exactly 2× (1,021,966 against 2,051,965), which is
the manipulated variable doing what it was told.

## What it says

**There is a crossover, somewhere between generation 100 and 200.** Doubling
self-play search is actively *harmful* early and clearly *helpful* later. Neither
arm is better; the question was malformed.

**The likely mechanism: search amplifies the value head, including its errors.**
Early the value head knows nothing, and a deeper search propagates its noise
further — at 64 sims the tree reaches 2.38 plies against 1.57 at 32
(`2026-10-07-what-search-is-worth.md`), so there is more bad evaluation between
the root and the backed-up number. The shallower search leans on the policy prior
instead, and at `sims=32, ra32` that prior gets a clean one-ply improvement over
all 32 root moves: `visits_each` is clamped at 1, sequential halving never
halves, and the result is exactly the regime Gumbel MCTS was designed for. Once
the value head is worth consulting, depth starts paying and the sign flips.

This is a prediction the data can be asked about later, not a demonstrated cause.
What is demonstrated is the crossover.

## What it retires

**My own framing from this morning.** I proposed this run as a gate: "if sims64
doesn't win on equal data, the per-hour version is almost certainly not worth
$13." The answer is that it depends entirely on when you ask, and a single
number for "does more self-play search pay" does not exist.

**And a probe that was 3.5× too large.** A 16-game calibration at gen400 came out
1W–14L–1D, about −394 Elo. The measured value at that same cell is −113 ±53. It
was reported as a probe and not a result, which is the only reason it did no
damage — but it is a clean demonstration of why 16 games is a smoke test. See
[[best-of-n-needs-replication]].

## Caveats

- **One training run per arm.** Training has seed variance that match error bars
  do not capture. The monotone shape across five pre-specified points
  (+136, +68, −139, −90, −113) is more persuasive than any single cell, but a
  replication would be a different and stronger claim.
- The crossover is bracketed between gen100 and gen200 and not located. A
  checkpoint at gen150 exists in both arms and would narrow it for ~20 minutes
  of match time.
- Measured at 128 match nodes. Whether the crossover moves with match budget is
  untested.

## The practical reading

If this holds, the setting is not a constant but a **schedule**: train at 32 and
raise it once the value head is worth searching against. That captures the cheap
arm's early efficiency and the expensive arm's late strength, and it costs
nothing to try — self-play sims is already a flag.

Before acting on it, see the wall-clock half below, since sims64 buys its late
advantage with 1.53× the time per generation.

---

# The wall-clock half, salvaged

`bench/sims_equal_wallclock.sh`. Raw: `data/2026-10-19-sims-equal-wallclock.json`.

This was recorded as unrecoverable. For the run as a whole it is: after the
sims32 arm wedged at **6.04h**, sims64 had the card to itself and its later
generations ran faster than contention would have allowed. Those are
contaminated.

**But below 6.04h both arms were running concurrently**, which is exactly the
symmetry the design required. Three pairs of saved checkpoints sit under that
line, matched on `elapsed_total` to within 6–13 minutes.

| matched wall clock | pairing | Elo | score |
|---|---|---|---|
| ~1.5h | s32-gen100 vs s64-gen50 | **+325 ±81** | 86.7% |
| ~2.8h | s32-gen200 vs s64-gen100 | **+147 ±52** | 70.0% |
| ~5.0h | s32-gen350 vs s64-gen200 | **−28 ±50** | 46.0% |

**Per hour, sims32 wins decisively and the margin erodes with time.** The 1.53×
cost per generation more than eats sims64's equal-data advantage over this
range: at equal data sims64 was ahead by 114 Elo late, and at equal time it is
still behind or level everywhere we can see.

The residual mismatches cut against the trend rather than for it: at the two
early rungs sims32 had 7–8% *more* clock, at the 5h rung sims64 had 3% more. So
the slope here is, if anything, slightly steeper than the truth.

## The decision, and where it stops

**For any training run up to about 5 hours, use `sims 32`.** That is measured,
it is a large effect, and it covers most of what gets run on this laptop.

**Beyond that it is unknown, and the honest answer is that we cannot see it.**
The three points are strikingly log-linear — slope −704 Elo per decade of wall
clock, residuals within ±5 — which puts the crossover near **4.5h** and would
predict sims64 ahead by roughly 300 Elo at 12h. That number should not be
believed:

- Three points against two parameters is **one degree of freedom**. Tiny
  residuals are close to guaranteed and are not evidence.
- Extrapolating 2.4× past the data is the exact move that failed this same
  morning: the leaf-depth trend looked just as clean over 128–512 and
  understated 2048 by 16%, because a schedule term came off its floor outside
  the measured range. See `2026-10-19-depth-at-the-top-of-the-budget.md` and
  [[derivation-is-not-measurement]].
- The real runs are 12–24h — `w-long` was 695 generations — so the regime we
  actually care about is entirely outside this measurement.

**This is now worth renting hardware for.** Not because the extrapolation is
convincing, but because it is *large and falsifiable*: it predicts sims64 ahead
by hundreds of Elo at 12h, where every measurement we have says sims32. A
12-hour matched rerun — **two GPUs, one arm per card**, per
[[two-cuda-processes-one-gpu-deadlocks]] — either confirms a schedule worth
adopting or kills the idea. The remaining ~$13 of credits covers it.

## What this changes about the deadlock

The note that a wedged arm destroys "per hour" questions beyond rescue was too
strong, and it cost a week of treating this as a dead end. The correct rule is
narrower: **a deadlock contaminates every comparison that involves the period
after it, and nothing before.** Checkpoints written while both arms were
competing are still wall-clock-matched by construction. Salvage first, then
decide what is lost.
