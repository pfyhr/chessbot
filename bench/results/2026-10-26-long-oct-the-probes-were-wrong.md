# The 72-hour run gained +145 Elo, and four probes said it gained nothing

Date: 2026-10-26. `runs/long-oct`: 72.01 h, generations 0–2601, M2 Max,
initialised from `runs/weekend/gen1371.pt`. Config: `--sims 32 --considered 32
--games 256 --concurrency 256 --steps 250 --batch 512 --lr 5e-4 --window 8
--blocks 6 --channels 96`. Raw: `runs/long-oct-eval/report.log`,
`runs/long-oct-fixed/report.log`.

## The result

**Published number: gen2600 beats weekend/gen1371 by +144.92 ± 50.73 Elo**, over
218 fixed-length games at 128 nodes, `RootActions 32`
(`runs/long-oct-fixed/report.log`).

The SPRT below reported +249.30 on the same pair. That estimate is biased
upward: SPRT stops the moment a boundary is crossed, which tends to be on a
favourable swing, so its point estimate is not an unbiased effect size. **SPRT
decides; fixed length publishes.** The decision is unaffected — the run gained a
great deal — but the magnitude was overstated by about 100 Elo, and this is the
second time on this project that a number read at its stopping point has had to
be revised down.

SPRT, bounds [0,50], logistic, alpha = beta = 0.05, 128 nodes, `RootActions 32`.
Positive Elo means the first-named checkpoint won.

| pair | Elo | games | verdict |
|---|---|---|---|
| gen2600 vs weekend/gen1371 | +249.30 ± 128.06 | 52 | H1 accepted |
| gen1300 vs weekend/gen1371 | +159.18 ± 96.23 | 70 | H1 accepted |
| gen2600 vs gen1300 | +94.68 ± 67.01 | 94 | H1 accepted |

The halves are consistent with the whole *within the SPRT readings* (159 + 95
against 249), which is a check of internal consistency only — all three share
the same upward stopping bias, so it does not validate the magnitude. What it
does establish is the shape: the second 36 hours were still clearly positive.
**The run had not plateaued at generation 2600.** Diminishing, not exhausted.

## What the probes said, and why every one of them was wrong

Block means over the run:

| gens | vs_random | vs_prev | mate | defend | tactics | policy_loss | value_loss |
|---|---|---|---|---|---|---|---|
| 0–199 | 0.799 | — | 0.023 | 0.390 | 0.207 | 2.335 | 0.366 |
| 800–999 | 0.815 | 0.500 | 0.022 | 0.335 | 0.178 | 2.309 | 0.370 |
| 1600–1799 | 0.843 | 0.504 | 0.011 | 0.290 | 0.150 | 2.260 | 0.396 |
| 2400–2599 | 0.845 | 0.499 | 0.012 | 0.293 | 0.152 | 2.244 | 0.384 |

Read together these say "2,601 generations bought nothing". The games say
+145 Elo. Taking them one at a time:

**`vs_prev` cannot detect this and never could.** It compares *consecutive*
generations. +145 Elo over 2,601 generations is about 0.06 Elo per generation —
indistinguishable from 0.500 at the per-generation scale, no matter how healthy
the run. A flat `vs_prev` is the expected reading of a run improving steadily.
It is not evidence of stagnation and must never be read as such again.

**`vs_random` is saturated.** At 0.80–0.85 it is near its ceiling, where large
strength changes move it very little and noise moves it as much.

**`mate`, `defend` and `tactics` are measured off-distribution.**
`train_chess.py:268` builds the suite once per process with
`ev.build_tactics(seed=11)` — a fixed set of positions found **by random play**.
As the policy sharpens on its own self-play distribution, random-play positions
become steadily more alien to it. A fixed random-play benchmark can therefore
decline while real strength rises, which is exactly what happened.

**`value_loss` rising is not a defect either.** Its targets come from the
network's own games, and as play strengthens the outcome distribution it has to
predict changes underneath it. The loss is against a moving target.

## The policy sharpened; it did not flatten

`bench/policy_entropy.py`, 2048 random-play positions (mean 30.5 legal moves):

| checkpoint | entropy | eff. moves | top-1 mass |
|---|---|---|---|
| weekend/gen1371 | 2.356 | 10.55 | 0.312 |
| long-oct/gen1300 | 2.271 | 9.69 | 0.335 |
| long-oct/gen2600 | 2.166 | **8.72** | **0.362** |

This kills the obvious rescue of the probe data — that the policy went flat and
started leaning on search. It did the opposite: top-1 mass rose from 0.312 to
0.362 and effective move count fell from 10.55 to 8.72, while search-on strength
rose 145 Elo.

## The search really does waste its budget — separately, and still worth fixing

This stands on its own and is not a diagnosis of the run, which succeeded.

`mcts.rs:522` sets `visits_each = (sims / (phases · m)).max(1)`, with `phases =
ceil(log2(m))` at line 490. At `considered = 32`: `phases = 5`, so phase one
takes `max(32/160, 1) = 1` visit on each of 32 root actions and **spends the
entire 32-simulation budget before the first halving.** Sequential halving never
runs.

The controlled sweep in `data/2026-10-07-leaf-depth.json` — one network
(gen695), one run, three widths:

| sims | `considered` | mean leaf depth | evals per move |
|---|---|---|---|
| 32 | 32 | 1.57 | 31.82 |
| 32 | 16 | 1.87 | 31.95 |
| 32 | 8 | **2.58** | **26.77** |

At `considered = 8` all three halving phases fit inside the budget. The result
is **64% more depth for 16% fewer evaluations** — cheaper and deeper, because the
narrow search can finish early when one action wins while the wide one always
spends everything.

Whether that trains better is a different question, and `bench/considered_ab.sh`
exists to answer it with games. Note that `2026-10-08-root-width-is-budget-dependent.md`
found `ra32` beats `ra8` for **match play** at 128+ nodes, and that still stands.
Playing well on a fixed budget and generating a good training target are not the
same objective.

## The lesson, which is the expensive part

Four of this project's metrics had already given confidently wrong answers. This
is the fifth occasion, and this time the conclusion was written before the games
were run — a diagnosis of a failure that had not happened, complete with a
mechanism. The mechanism was plausible, internally consistent, and wrong.

The rule that follows is narrower than "don't trust proxies", because the real
error was one of order: **the games were already queued. The write-up should
have waited for them.** A proxy may motivate a hypothesis. Nothing but a game
may close one.
