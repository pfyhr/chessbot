# Threading the driver by game: a measured zero

2026-10-26. The self-play driver was threaded at game granularity and measured
on Jupiter in three phases. It buys **1.27x on the driver in isolation, and
nothing at all on a real generation**. The rented-GPU plan has to assume a
single-threaded driver.

This was a prediction, written down here on 2026-08-20 and now wrong:

> games are independent, so the fix is threads rather than better single-thread
> code. Size it at about one thread per 300k evals/sec of accelerator.

The first clause is still true — games *are* independent, and the correctness
tests below prove the implementation exploits that cleanly. The second clause
does not follow from it, and the gap between the two is the whole result.

## What was built, and why it is not the 2026-10-06 attempt

On 2026-10-06 the loops inside `next_batch` were handed to rayon per slot and
measured slower: 4.8s to 5.4s at 4 threads (86b661a, reverted in 2131d58). A
call carries ~5ms of work spread over 256 slots, so per-slot dispatch cost more
than the work it handed out. The code comment left behind named shard
granularity as the prerequisite if it were ever retried.

So: slots are sharded into contiguous blocks, one per thread, each owning whole
games — its own trees, its own trajectories, its own rows of the batch. A call
dispatches `threads` tasks, not `concurrency` of them, on a pool built once at
construction. `threads=1` takes a path with no pool at all.

Two things had to move to make a slot self-contained, and the second was a
latent bug worth recording independently of the performance result:

- Finished games and the win/draw/ply tallies moved onto the slot, summed on read.
- **Refills now come from a per-slot game quota instead of a shared counter.**
  Whichever slot finished first used to take the next game, so game-to-slot
  assignment depended on completion order. Harmless while sequential; silently
  irreproducible the moment it is not. In the training config
  (`concurrency == games`) every quota is zero, so this changes nothing there.

The reproducibility test is stronger than "same seed and K, identical games":
K=1 against K=2,3,4,8, compared ply by ply on the policy targets, with and
without PCR, at a concurrency that forces the refill path. The per-slot RNG
(`seed ^ k * 0x9E3779B97F4A7C15`) makes slot order irrelevant by construction;
the quota removes the last completion-order dependence.

## Phase 1 — the driver alone, no network

`drivercost` feeds an instant evaluator, so this is tree descent and encoding
with the GPU removed entirely. `--sims 32 --concurrency 512 --games 2048`.

| threads | evals/sec | speedup | efficiency |
|---:|---:|---:|---:|
| 1 | 314,542 | 1.00x | 100% |
| 2 | 359,141 | 1.14x | 57% |
| 4 | 400,285 | 1.27x | 32% |
| 8 | 377,506 | 1.20x | 15% |

Efficiency roughly halves at every doubling, and 8 threads is *worse* than 4.
This is the best case the change will ever see — perfect independence, no
accelerator, no contention — and it is 1.27x.

## Phase 2 — driver seconds per generation, with a real evaluator

The number that transfers between machines is driver seconds per generation,
not wall clock. `generation_profile.py --games 128 --concurrency 128 --sims 32
--considered 32 --steps 100`, no `torch.compile` (variable batch shapes would
recompile per shape; compile only touches the GPU half anyway).

| threads | driver s/gen | generation total | GPU-bound |
|---:|---:|---:|---:|
| 1 | 6.4 s | 83.5 s | 92.3% |
| 2 | 5.9 s | 82.2 s | 92.8% |
| 4 | 6.1 s | 82.4 s | 92.6% |

Flat. A 0.5 s spread on unreplicated runs, with 4 threads worse than 2 — the
honest reading is no effect, not "2 is optimal".

**Phase 1's 1.27x does not survive a real evaluator.** The likely reason is
visible in the configs: phase 1 ran 512 slots, phase 2 ran 128. A quarter of the
work per thread, amortising the same dispatch. The 2026-10-06 failure was
granularity; this is the same arithmetic one level up.

## Phase 3 — real training, compile on, alternating blocks

`chess-train` with `--compile`, 5 generations per block, gen 0 dropped (it pays
compilation) and evaluated generations excluded. A-B-A-B and strictly
sequential, never concurrent.

| block | s/gen | mean plies |
|---|---:|---:|
| threads-1-1 | 90.8 | 130.4 |
| threads-4-1 | 85.8 | 130.4 |
| threads-1-2 | 79.5 | 130.4 |
| threads-4-2 | 83.8 | 130.4 |

Identical `mean_plies` across all four, so the arms did identical work.

Pooling the 6 clean generations per arm:

| arm | mean | sd | min | max |
|---|---:|---:|---:|---:|
| 1 thread | 85.18 s | 9.42 | 74.5 | 99.0 |
| 4 threads | 84.80 s | 4.09 | 80.0 | 89.9 |

**Difference +0.38 s, or +0.5% ± 9.8%** (95% CI −8.0 to +8.8 s). The error bar
is twenty times the effect.

## The 97.0 s baseline was measuring something else

The benchmark was written to compare against `long-oct`'s 97.0 s/gen on this
machine. **Today's single-threaded baseline is 85.2 s/gen.** Quoting the
4-thread 84.8 s against 97.0 would have reported a 13% win that the *one-thread*
arm earns entirely by itself.

The cause is not drift. Every one of `long-oct`'s 2,602 generations carries
`evaluated: true` — the 97.0 s **includes the per-generation probe suite**
(`vs_prev`, `vs_random`, `mate`, `defend`, `tactics`, `material`). This
benchmark set `--eval-every 100000` and filtered on `evaluated == false`
precisely to keep probes out of a timing measurement. The two numbers were never
comparable:

| | self-play + train | probes | total |
|---|---:|---:|---:|
| `long-oct`, as recorded | ~85 s | ~12 s | **97.0 s** |
| this benchmark | **85.2 s** | excluded | — |

The configurations otherwise match: 256 games, `mean_plies` 130.4 here against
`long-oct`'s 128.7.

**This has already propagated into the rental decision criteria.** Both
`docs/rented-training-plan.md` and the threshold block in `bench/vast_deploy.sh`
state "Jupiter is 97.0 s/generation, 37.1 generations/hour" and derive the
worth-renting and kill thresholds from it — while the rented box is measured
with `generation_profile.py`, which reports `generation total (no eval)`. A
no-probe number against a with-probe baseline **overstates every candidate box
by about 14%**, and it is enough to flip a verdict:

> A box at 45 s/gen reads as 2.16x against 97.0 and survives the kill
> criterion. Against the correct 85.2 s it is 1.89x and is killed.

Corrected thresholds, on the like-for-like baseline of 85.2 s/gen = 42.3 gens/h:

| | was (vs 97.0) | now (vs 85.2) |
|---|---|---|
| worth renting, >= 3.0x | >= 111 gens/h, <= 32.3 s | **>= 127 gens/h, <= 28.4 s** |
| kills it, < 2.0x | < 74 gens/h, >= 48.5 s | **< 85 gens/h, >= 42.6 s** |
| `drivercost` 1-thread floor | < 120,000 evals/s | **< 126,000 evals/s** |

The last row moves because the floor was defined as 0.4x Jupiter, and Jupiter's
single-thread figure is now measured at 314,542 rather than the 292,646 of
2026-08-20.

This is the same error class as [the 6.3x GPU ceiling that was the borrowed
machine's CPU](2026-10-06-the-ceiling-was-the-other-machines-cpu.md) — a number
from one run read against a number from another, where the two were not
measuring the same work. It was caught only because the design alternated arms
within a single session, which made the 1-thread arm its own control.

## Two things the data shows that were not the question

**The serial path got faster.** 314,542 evals/sec against 292,646 measured on
2026-08-20 on this machine, ~7%. Plausibly the removed shared counter and the
slot-local bookkeeping; equally plausibly two months of machine and toolchain
drift. Not separable from this data, and not claimed.

**The 4-thread arm is less variable.** sd 4.09 against 9.42, consistent across
both blocks. n=6 per arm, so this is an observation and not a finding.

**Every block's middle clean generation is the slowest** — 99.0, 89.9, 89.1,
89.6 against 74.5–84.3 on either side of it. Four for four, in both arms, so it
is systematic rather than noise. Unexplained, and out of scope here, but it is
the largest single structure in the timing data and it is not thermal.

## What this does to the rented-GPU case

The case was never Jupiter, where the GPU is ~92% of the generation and no
driver win could matter. It was the predicted 18.7 s driver against 13.6 s GPU
on a rented 4090 — 58% of the rental spent on one core doing tree descent.

Even granting that estimate, a 1.27x ceiling takes 18.7 s to 14.7 s: about 12%
off the generation, not the 2x the split invites. And 1.27x is precisely the
number that failed to reproduce once a real evaluator was attached.

Two caveats that cut in opposite directions, and neither is resolved here:

- A rented box has more cores and a different memory system than an M2 Max.
  Phase 1's scaling is a property of this machine's cache hierarchy as much as
  of the code.
- The 18.7 s driver estimate is *itself* an estimate, derived rather than
  measured, and the measured driver numbers here are from a 128-slot profile
  config rather than the training config.

**Plan on a single-threaded driver.** If the rented box is measured and the
driver really is 58% there, phase 1 can be re-run on it in two minutes to see
whether 1.27x travels.

## Reproducing

```
bench/driver_threads.sh                   # ~37 min, defaults
GENS=3 BLOCKS=1 bench/driver_threads.sh   # ~10 min, still paired
```

Raw: `data/2026-10-26-driver-threads.log` and the four `history.json` files
beside it.
