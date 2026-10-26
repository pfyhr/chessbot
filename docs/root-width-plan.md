# Plan: what `--considered` should be, and how to find out cheaply

Written 2026-10-26, after `runs/long-oct`. Supersedes nothing; it is the
follow-on to `docs/big-run-plan.md`.

## The finding that prompted this

Gumbel AlphaZero's Sequential Halving splits `n` simulations into `log2(m)`
phases and visits every surviving contender equally often — `⌊n / (log2(m)·m)⌋`
per phase (Danihelka et al. 2022, Figure 1). For that floor to be at least 1:

> **n ≥ m · log2(m)**

`runs/long-oct` trained 72 hours at `--sims 32 --considered 32`, where that
quantity is 160 against a budget of 32. The consequence, measured and derived
two independent ways:

| m | m·log2(m) | vs n=32 | halving phases completed | predicted sims used | measured evals/move |
|---|---|---|---|---|---|
| 4 | 8 | ok | 2 of 2 | 32 | — |
| 8 | 24 | ok | **3 of 3** | 26 | **26.77** |
| 16 | 64 | over | 3 of 4 | 32 | 31.95 |
| 32 | 160 | over | **1 of 5** | 32 | 31.82 |

The predicted column is the paper's arithmetic; the measured column is
`data/2026-10-07-leaf-depth.json`, instrumented on gen695. They agree. Measured
leaf depth at `sims 32`: **1.57 ply at m=32, 1.87 at m=16, 2.58 at m=8**.

**At 32 simulations, m=8 is the only width where sequential halving completes.**
`Config::default()` in `mcts.rs` says `max_considered: 16`, which is the paper's
default for the paper's budgets (it needs n ≥ 64). We inherited the constant
without the budget that makes it valid, then overrode it to 32.

## The constraint that shapes everything: effect size

long-oct gained **+144.92 ± 50.73 Elo** over 2,601 generations (218 fixed-length
games at 128 nodes). Near the end of the run that is ~0.073 Elo per generation.
So the gap between two arms, if one learns faster:

| generations per arm | +10% | +20% | +30% |
|---|---|---|---|
| 300 | 2.2 | 4.4 | 6.6 |
| 1000 | 7.3 | 14.6 | 21.9 |
| 1500 | 11.0 | 21.9 | 32.9 |

Resolving ~20 Elo takes roughly 1,500 games. So a naive three-arm training A/B
is 1,500 generations × 3 arms ≈ **5 days of compute before a single game is
played** — and renting hardware does not change that arithmetic, it only moves
the cost from time to money.

The plan is therefore built around *not needing that run*.

## Step 1 — Measure target quality directly, without training (free, ~1 h)

What `--considered` changes is the quality of the **training target**, and that
is measurable without training anything.

- Sample ~1,000 positions from **gen2600's own self-play**. On-distribution by
  construction: long-oct's tactics probes misled us precisely because they are a
  fixed set drawn from random play (`train_chess.py:268`, `build_tactics(seed=11)`),
  which drifts further from the policy as the policy improves.
- Reference target: `n=2048, m=32`, measured depth 11.63 ply — the best estimate
  of the right answer we can produce.
- Candidates: `n=32` at m ∈ {8, 16, 32}.
- Score: `KL(pi_ref || pi_cand)` and top-1 agreement.

Cost is trivial: the reference is ~2M evaluations, about 4 minutes at the
measured 10,700 evals/s. Candidates are seconds each.

## Step 2 — Validate the proxy against a known training outcome (free, same hour)

Five of this project's metrics have given confidently wrong answers, most
recently to this author, so the proxy is not trusted until it earns it.

`bench/results/2026-10-21-sims-settled.md` measured `sims 64` as better per game
than `sims 32` by −239 Elo at equal data. Run the identical KL metric on
`n=32` against `n=64`. **If it does not rank them in that order, discard the
proxy and go straight to training.** Also check the ranking is stable with
`n=1024` and `n=4096` references, since "deep search is truth" is an assumption.

## Step 3 — One training confirmation, two arms (not three)

Steps 1–2 reduce {8, 16, 32} to one challenger. Then a single A/B:

- both arms initialised from `runs/long-oct/gen2600.pt`
- `--sims 32`, differing only in `--considered`
- ~1,500 generations per arm
- judged **head-to-head**, ~1,500–2,000 games, fixed length, not SPRT

Fixed length matters here. On this very question SPRT reported +249.30 on 52
games where 218 fixed-length games gave +144.92 — SPRT stops at a boundary
crossing, which biases the point estimate upward. SPRT decides; it does not
publish.

Report both equal-generation and equal-wall-clock readings, reconstructed from
`history.json` per arm as `bench/sims_curves.py` does. The arms differ in cost
(26.77 against 31.82 evals/move), so the two readings will not agree.

## Where to run it

**Quality per generation is hardware-independent** — only the resulting
checkpoint matters. Cost per generation is measured separately, in short
alternating blocks on one machine, exactly as `bench/compile_ab.sh` did. That
decoupling means the two arms may run on *different* machines at the same time.

| option | wall clock | cost | note |
|---|---|---|---|
| **Jupiter + aida in parallel** | ~40 h | free | aida is borrowed: gpu0 only, ask first |
| Jupiter alone, sequential | ~80 h | free | shares CPU with FEA |
| rented 2-GPU box | ~25 h | ~$20–30 | only helps if the loop is GPU-bound |

Recommendation: **do not rent.** Steps 1–2 are free and may collapse the
question. Renting also needs an explicit decision, since the standing limits say
no spending.

## On rewriting the loop to run fully on a GPU

Not now, and probably not at all.

MCTS tree descent is branchy pointer-chasing with sequential dependencies —
close to the worst case for a GPU. The present design, a Rust driver batching
leaf evaluations across 256 concurrent games into one network call, is already
the standard architecture for getting utilization out of this workload.

More to the point, there is no evidence the tree work is the bottleneck. The
`hotprof` build exposes `ns_*` counters; one profiling run gives the split. If
network evaluation dominates, a faster GPU helps and a rewrite is pointless. If
tree work is under ~30%, Amdahl caps what a rewrite could win. **One measurement
decides it before anyone writes code.**

## Open items this plan does not cover

- **The σ scale may matter more than the width.** The paper uses
  `c_visit = 50, c_scale = 1.0` (Eq. 8). We use `c_scale = 0.1` against Q
  rescaled to [0,1], so our σ spans ~5 where the paper's spans ~100 — the
  search moves the training target roughly an order of magnitude less than
  intended. The code comment defends this, but the behaviour it avoids is what
  the paper describes as the point ("reduces the effect of the prior policy").
  Neither constant is a CLI flag; exposing them is a small change worth making
  while the bridge is open.
- **`chess_eval.py:213` hardcodes `max_considered=16`** for the in-training
  search probe, so that probe has never reflected the `--considered` the run was
  trained with.
- **The 1-node match was never run.** ~10 minutes. It separates "the raw policy
  got weaker" from "the probes are off-distribution" and is the cheapest open
  question in the project.
