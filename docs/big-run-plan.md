# Plan: the long training run

Status: **draft, 2026-10-21.** Three decisions are the user's and are marked
**[DECIDE]**. Everything else is settled by measurement and cited.

## Why now

Every tuning question that could change the configuration has been answered.
Nothing further is worth measuring before committing hours to training:

| question | answer | source |
|---|---|---|
| self-play simulations | **32** | `2026-10-21-sims-settled.md` |
| self-play root width | **32** (96% of 24.2 mean legal moves) | `2026-10-07-split-root-width.md` |
| match root width | **32** at every budget we use | `2026-10-10-width-budget-sweep.md` |
| network size | **6×96** | `2026-10-06-capacity-three-sizes.md` |
| what search is worth | +539 Elo, 32→1024 nodes | `2026-10-07-what-search-is-worth.md` |

## The finding this plan is built on

Two independent "spend more per step" knobs were tested, and both produced the
same shape:

| knob | at equal data | at equal wall clock |
|---|---|---|
| bigger net (8×128 over 6×96) | +80 … +173 Elo | peaks +64, **ends −21** |
| more sims (64 over 32) | −239 mean, −402 peak | peaks −80, **decays to −37** |

Spending more per step reliably wins on data and reliably fails to keep that win
once time is priced in. The equal-time curve peaks mid-run and turns back down in
both cases. **Wall clock is the binding constraint**, so the long run takes the
cheapest configuration that trains well and spends everything on duration.

That is a prediction as much as a conclusion: it says a third knob of the same
kind would behave the same way. Worth remembering if a future "obvious upgrade"
is proposed.

## Configuration

```
--blocks 6 --channels 96        11.0M params
--sims 32 --considered 32       self-play search
--games 256 --concurrency 256   per generation
--steps 250 --batch 512         per generation
--lr 5e-4 --window 8            optimizer, replay window
--max-plies 400 --eval-every 1
--keep-every 25                 see checkpointing below
```

Resume from **`runs/weekend/gen1371.pt`** — the strongest network we have at
54.1h and `vs_random` 0.850 — rather than from scratch. Optimizer state is
preserved in checkpoints written after 2026-08-25, so a resume really continues.
Resume into a **fresh directory**: `train_chess.py` starts `history = []` and
`range(0, …)`, so resuming in place overwrites the record.

## [DECIDE] 1 — where it runs

| option | cost | notes |
|---|---|---|
| **aida gpu0** | free | Stefan's box. Ask before scheduling; keep to gpu0. |
| rental, 1×4090-class | ~$0.55/h | ~$40 for 72h |
| rental, 2×4090 | ~$0.95/h | only worth it for a second arm |

Recommendation: **aida gpu0 if Stefan can spare it**, since this is a single-arm
run with no need for a matched second card, and the cost is then zero. A rental
is the fallback. Do not rent two cards for one arm.

## Running: `runs/long-oct`, started 2026-10-22 22:32 CEST

Resumed from `weekend/gen1371` on Jupiter, 72h, ending ~Sun 25 Oct 22:32 CEST.
tmux session `train`; checkpoints local at keep-every 25.

**Measured 95s/generation, against the 142s this plan projected** — 1.5x faster,
so ~2,719 generations rather than ~1,825. Checked for the obvious confound and
it does not hold: the weekend run was flat at 138-143s across all 1,372 of its
generations at the same mean game length (~130 plies), so this is not a warm-up
or game-length effect. The cause is software — newer torch, `torch.compile` on
by default, and bridge work since September.

**`train.log` is useless for monitoring.** Python buffers stdout to a file, so
it sat at 109 bytes reporting zero generations while the run was 15 generations
in. Read `history.json` or the checkpoint mtimes instead. This is the same
failure as the watchdog regex that cost 10h on the rental: a status channel that
does not update is worse than none, because it reads as authoritative.

## [DECIDE] 2 — how long

The longest run so far is `weekend`, 1,372 generations over 54.1h. Meaningful
options:

| target | generations | what it tests |
|---|---|---|
| 72h | ~1,800 more | does the curve keep climbing past 1,372? |
| 1 week | ~4,200 more | first run long enough to plateau visibly |

Recommendation: **72h first**, with checkpoints that make a longer continuation
free. A week is only worth committing once 72h shows the curve still rising.

## [DECIDE] 3 — scope of the evaluation

Minimum (required): booked-opening matches against the previous champion at
fixed nodes, plus the `lc0` 16×2 anchor at 1 node, which currently scores 62.5%
against `weekend/gen1371`.

Optional, and currently item 2 of STATUS.md's list: **SPRT instead of
fixed-length matches.** Every Elo figure in this project is a 50–300 game match
read at face value. `fastchess` supports SPRT and it is installed. Worth doing
once, as a methodology upgrade, but it is not a prerequisite.

## Checkpointing and retrieval

Checkpoints are 126 MB. At `--keep-every 25` a 72h run writes ~72 of them,
about 9 GB.

- **Rent ≥50 GB of disk.** The default is 16 GB and it is the usual binding
  constraint. See [[vast-disk-is-the-hidden-constraint]].
- **Copy each checkpoint off as it lands**, in a loop beside the run — not in
  one transfer at the end. On 2026-10-21 a host advertising 857 Mbps delivered
  0.2 Mbps to Sweden, and 4.4 GB took a rescue operation to retrieve. The cost
  is the same bytes spread over the run instead of a lump at the point of
  greatest regret. See [[pull-checkpoints-inline]].
- **Measure egress during deployment.** `vast_deploy.sh` now does this and
  refuses to recommend a box that cannot return its own results.

## Monitoring

- Watch **progress, not liveness**: generation counter and log mtime, sampled
  twice. On 2026-10-09 a wedged arm passed every liveness check for 9.7h.
- The watchdog in `sims_ab.sh` now requires CPU corroboration from
  `/proc/<pid>/stat` before killing anything. It previously killed two healthy
  arms at 13.6h of a 24h run because its generation regex stopped matching at
  gen 1000.
- **Check in on a schedule.** The 10.4h of lost training above cost a further
  ~11h of idle billing purely because nothing reported the stop. A single daily
  look would have caught it.

## What would make this run a failure, and what to do then

- **The curve plateaus early.** Then the question becomes what limits it —
  capacity, data diversity, or the replay window — and that is a new study, not
  a longer run.
- **It diverges or collapses.** Resume from the last good checkpoint; this is
  why `--keep-every 25` is worth the disk.
- **It merely matches `weekend/gen1371`.** That is still informative: it would
  say 54h is near the useful limit of this configuration, which is the strongest
  argument yet for changing something structural.

## Not in scope

`ra64` at 1024 nodes (the untested upper end of the width sweep, ~4% more move
coverage) and the `--init` replay-buffer warm-start. Both are small, and neither
blocks this run.
