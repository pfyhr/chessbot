# Status

_Last updated 2026-10-26. **Nothing is running locally — Jupiter is free for
other work.** Next step is `docs/root-width-plan.md`, steps 1 and 2, which are
free and take about an hour._

## 2026-10-26 — long-oct finished, and the probes were wrong about it

`runs/long-oct`: 72.01 h, 2,601 generations from `runs/weekend/gen1371.pt`.

**It gained +144.92 +/- 50.73 Elo** (218 fixed-length games, 128 nodes).
Its own probes said it gained nothing: `vs_prev` sat at 0.500 all run,
`vs_random` drifted inside its noise, and `mate`/`defend`/`tactics` all fell by
a quarter to a half. Every one of those readings is explicable and none of them
means stagnation -- see
`bench/results/2026-10-26-long-oct-the-probes-were-wrong.md`.

Two numbers needed correcting on the way, both in the same direction:
- SPRT reported +249.30 on 52 games; 218 fixed-length games gave +144.92.
  SPRT stops at a boundary crossing and its point estimate is biased upward.
  **SPRT decides, fixed length publishes.**
- A diagnosis of "self-distillation fixed point" was written from the probes
  before the games were run, and the games refuted it.

**`torch.compile` is worth 1.36x** (97.0 against 131.9 s/gen, four alternating
blocks), about 75% of the 1.48x gap to the September weekend run --
`bench/results/2026-10-26-compile-is-most-of-the-speedup.md`.

**The search wastes most of its budget.** At `--sims 32 --considered 32`,
Sequential Halving needs `n >= m*log2(m)` = 160 and has 32, so phase one
consumes everything and no halving ever runs. Measured depth 1.57 ply; at
`--considered 8` all three phases fit and depth is 2.58. Plan in
`docs/root-width-plan.md`.

The raw policy got **stronger**, not weaker: at 1 node, where there is no
search and the move is the policy's own, gen2600 beats gen1371 by
**+192.40 +/- 34.73** over 300 games -- a larger margin than the +144.92 the
same pair shows at 128 nodes. The `mate`/`defend`/`tactics` decline was
distribution shift in a fixed random-play suite, nothing more.

## Next: establish a standing evaluation protocol

The anchor problem is already solved -- see
`bench/results/2026-09-25-first-calibrated-opponent.md`. Lc0 with a 16x2 net at
1 node is the first opponent neither crushed nor crushing (gen1371 scored
37.5%). With gen2600 at +145, the projected ladder is 58% / 38% / 16% at lc0
1 / 4 / 16 nodes -- three usable rungs. `lc0` is installed and both nets are in
`runs/nets/`.

What is missing is that the protocol lives in a results file rather than a
script. Codify `bench/gauntlet.sh`: frozen opponent set, **fixed nodes never a
clock** (a contended match read +137 +/-49 where an idle one read +241 +/-73 on
the same pair), booked openings, ~400 games per rung, and a maximum-likelihood
Elo fit over the whole gauntlet rather than chained pairwise deltas.

## Open and cheap

- **`c_scale` is 0.1 where the paper uses 1.0**, so the search moves the
  training target about a tenth as much as intended. Not a CLI flag yet.
- **`chess_eval.py:213` hardcodes `max_considered=16`**, so the in-training
  search probe has never matched the trained `--considered`.

## 2026-10-21 — self-play search: better per game, barely better per hour

**`sims 32` stays the self-play default.** A 13.6h two-GPU A/B (one arm per
card) measured both halves of the question on 1,900 games:

- **Equal data:** doubling search is worth a mean **−239 Elo** from gen200 on,
  peaking at **−402 ±123**. The largest effect any single setting has shown in
  this project.
- **Equal wall clock:** that collapses to at most **−80 ±56** around 7.5h, and
  *decays* to −37 by 12.66h. `sims32` wins outright below ~4h.

The gap between those two rows is the whole answer: `sims64` costs **1.76×** per
generation, so at a matched hour it has seen half the games, and the quality
gain very nearly cancels. One point (~4.9h, −28 ±54) reproduces the 19 Oct run's
−28 ±50 on different hardware, which is the best replication this project has.

Four extrapolations along the way all overstated the per-hour benefit, by 2.6×
to 8.5×, because the curve has a peak rather than being log-linear. See
`bench/results/2026-10-21-sims-settled.md` and
[[self-play-search-has-a-crossover]].

**Cost:** the run should have been 24h. A watchdog bug of mine — a regex that
stopped matching once the generation number reached four digits and lost its
leading space — killed two healthy arms at 13.6h, and ~11h of idle billing
followed because nothing reported the stop. Fixed, with CPU corroboration
required before any kill.

## 2026-10-19 — search depth at the top of the budget, and the sims A/B settled

**Leaf depth measured at 1024 and 2048 simulations**, the two budgets the
10 Oct width sweep reasoned about without ever measuring: **8.95 and 11.63
plies** at `ra32` (plies, not moves — about 4.5 and 5.8 moves each side).

The published explanation holds. Narrowing the root buys depth, and depth only
pays while it is scarce; at 1024 nodes `ra32` on its own already puts 42% of its
leaves at depth 9 or beyond, so narrowing there buys something the search has.
`ra32` stays the default. The two blank cells in the sweep table are now real
numbers.

What the measurement *broke* is the rate, which matters more than it sounds.
"Depth grows about 1.3 ply per doubling" was true over 128–512 and false outside
it: growth jumps to **2.4–2.7 ply per doubling** at the top, because
`visits_each = max(sims/(phases*m), 1)` comes off its floor of 1 and consecutive
simulations begin descending an already-carved line instead of widening a
shallow frontier. My own extrapolation from that trend predicted 8 and 10 and
was 16% low. A schedule artifact, not a law. See
`bench/results/2026-10-19-depth-at-the-top-of-the-budget.md`.

**The sims 32 vs 64 self-play A/B is answered, both halves.** At equal data the
cheaper arm is **+101 ±37** early (gen 50–100) and **−114 ±30** late
(gen 200–400) — the sign flips between gen100 and gen200, so doubling self-play
search is actively harmful early and clearly helpful later. At equal wall clock
`sims32` wins **+325 ±81** at 1.5h, **+147 ±52** at 2.8h and is level at 5.0h;
the 1.53× cost per generation more than eats the late advantage everywhere we
can see.

**Decision: `sims 32` for any training run up to ~5h.** Beyond that is unknown.
See `bench/results/2026-10-19-sims-crossover.md` and
[[self-play-search-has-a-crossover]].

**The 9 Oct deadlock was less destructive than recorded.** It contaminates only
the period *after* the wedge; both arms shared the card up to 6.04h, so every
checkpoint pair below that line was wall-clock-matched by construction. That is
where the equal-wall-clock curve came from, after a week of treating the
question as dead.

### Next
A **12h matched rerun, two GPUs, one arm per card**. The three wall-clock points
are log-linear (−704 Elo per decade) and would put `sims64` ~300 Elo ahead at
12h, which is the regime we actually train in — but that is three points against
two parameters extrapolated 2.4×, the same move that just failed on leaf depth.
Large and falsifiable, so worth the remaining ~$13.

---

## 2026-09-25

_Two results landed today.

**The policy-head search is over, and the incumbent won.** A 1000-game round robin
(400 per engine) ranks `A-flat` +112 ±32, ahead of the `pw32-g145` anchor at +39,
with `B-conv` −13, `C-narrow` −21 and `D-attn` −118. Three architectural
rearrangements have now failed to beat the 6×96 flat-head network. The
"global mixing" explanation is refuted — see
`bench/results/2026-09-25-policy-head-round-robin.md`.

**There is a calibrated opponent for the first time.** `lc0` with a 16×2 net
(`runs/nets/11258-16x2-se-4.pb.gz`) at **1 node** scores 62.5% against
`weekend/gen1371` — i.e. we score 37.5%, squarely inside a measurable band, where
Stockfish and every strong net saturate. Use **fixed nodes, not a time control**:
both engines honour `go nodes`, which makes matches reproducible on a busy
machine. See `bench/results/2026-09-25-first-calibrated-opponent.md`._

## Where it stands

A working AlphaZero-style loop, trained for about **77 hours total** on one M2 Max.

**Strongest network: `runs/weekend/gen1371.pt`** — **+220 ±95 Elo above `night/gen248`**
(booked ladder, 2026-09-20), which is itself ~816 above a random mover. Do not add those
two numbers and publish the total: see `bench/results/2026-09-20-weekend-resume.md` for why
the random-mover anchor is saturated and the sum is not a measurement.

It still plays a real opening and then hangs pieces; a human who knows the rules beats it.

```
.venv/bin/chess-uci                  # UCI engine — point any GUI at this
.venv/bin/chess-serve                # built-in browser board
.venv/bin/chess-play --ladder        # terminal client, staircase mode
```

`chess-uci` and `chess-serve` auto-select the most recently trained run, so both currently
load `runs/night`. Generation is a UCI option, which makes any GUI a strength ladder.

## What is built

| | |
|---|---|
| `crates/az-core` | `Game` trait, chess + Connect4, Gumbel MCTS with proven-win propagation, batched self-play driver |
| `crates/az-py` | PyO3 bindings — `chessbot_core` |
| `python/chessbot` | network, training loop, evaluation gates, UCI engine, browser board, terminal client |
| `bench/results/` | every measurement that drove a decision, including the ones that were wrong |

46 Rust tests, 70 Python tests, clippy clean.

## Measurement rules, learned expensively

Four separate times a measurement sent this project after a problem that did not exist.
Both rules below cost days.

1. **When a proxy disagrees with head-to-head play, the proxy is wrong.** Material, loss
   curves and tactical suites are worth watching; none of them is the objective. Three
   proxies once agreed that a resume was degrading the network. It was not.
2. **A match without an opening book is not a measurement.** Near-deterministic engines
   from a fixed start play one game, however many times you run it. This understated the
   engine's rating by more than 2× and produced one result of exactly 10W–10L with a
   ±0.00 error bar. Use `bench/openings-4ply.epd`.

## Known problems

- **Value head is weak.** It cannot separate winning a pawn from hanging a queen; every
  opening move evaluates to 46–50%. This is the main thing holding play back — the search
  is carrying the engine.
- **Raw policy is poor** and search corrects it: 1.g4 at 61% raw becomes 1.e4 at 69% after
  search. Fine for playing, but it means the shortlist Gumbel samples from is close to
  arbitrary, which is why widening `RootActions` from 16 to 32 was worth +102 Elo
  (`pw-32` vs `pw-16`, head to head; the +160 sometimes quoted is against the baseline
  and bundles proven-win propagation with the width change).
- **`--init` resumes weights but the buffer starts empty.** Optimizer state is now saved,
  but every checkpoint written before 2026-08-25 lacks it and will warn on resume.
- **Mate-in-1 on raw policy sits near chance** (0.065 against a 0.031 baseline) even though
  the *engine* finds mates reliably. The suite is drawn from random walks and is a hard
  out-of-distribution probe; it may simply be unreachable at this scale.

## Next, roughly in order of expected value

1. ~~**Sweep `RootActions`.**~~ **Done 2026-10-10** — a 16-cell sweep over width × node
   budget, 1,800 games. There is no single best width: the optimum walks right with the
   budget and saturates at `ra32`, which already covers 96% of the 24.2 mean legal moves.
   `ra8` is worth +41 ±25 at 128 nodes and loses everywhere above it. **Default stays
   `ra32`.** See `bench/results/2026-10-10-width-budget-sweep.md`. The mate suite's hint
   that it "does not saturate until 64" was not borne out.
2. **SPRT instead of fixed-length matches.** Every Elo figure here is a 50–200 game match
   read at face value. `fastchess` supports SPRT and it is already installed.
3. **Longer runs from `runs/night/gen248.pt`**, now that resume keeps optimizer state.
4. **Wall-clock time-ladder checkpoints** (Phase 4's remaining half) — snapshots on a
   logarithmic schedule so "how good was it after ten minutes?" is playable.
5. **A bigger network.** 6×96 is 11M params, sized for a loop that had to be debugged
   quickly rather than for strength.

## Unused resources

An M1 Mac is available. The measured bottleneck is GPU, not CPU — the self-play driver
supplies 292k evals/sec on a single thread while the GPU consumes 10–24k — so a second
machine is worth more for running experiments in parallel than for federating one run.
Renting is cheaper still: ~$110 of 4090 time is roughly AlphaZero's entire 44M-game run,
see `bench/results/2026-08-20-compute-budget.md`.
