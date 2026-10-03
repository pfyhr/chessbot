# Status

_Last updated 2026-09-25. **Nothing is running; the machine is idle.**

Two results landed today.

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
  arbitrary, which is why widening `RootActions` from 16 to 32 was worth +160 Elo.
- **`--init` resumes weights but the buffer starts empty.** Optimizer state is now saved,
  but every checkpoint written before 2026-08-25 lacks it and will warn on resume.
- **Mate-in-1 on raw policy sits near chance** (0.065 against a 0.031 baseline) even though
  the *engine* finds mates reliably. The suite is drawn from random walks and is a hard
  out-of-distribution probe; it may simply be unreachable at this scale.

## Next, roughly in order of expected value

1. **Sweep `RootActions`.** 16 → 32 was worth +160 Elo and it has never been tuned. The
   mate suite suggests it does not saturate until 64.
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
