# torch.compile is worth 1.36x, and explains three quarters of the speedup

Date: 2026-10-26, local M2 Max, `bench/compile_ab.sh`, torch 2.13.0 / Python
3.14.3 / MPS. Raw: `runs/compile-ab/report.log`.

## The question

`runs/long-oct` averaged 95 s/generation. The weekend run of 2026-09-20, same
config and the same mean game length (~130 plies), averaged 141 s and was flat
at 138–143 s across all 1,372 of its generations — so the 1.48x gap is not
warm-up and not game length. It is software: a newer torch, `torch.compile` on
by default since, and bridge work in between. This isolates the compile half.

## The measurement

Four blocks of 6 generations, alternating, run **sequentially**. Generation 0 of
each block is discarded. Rates read from `history.json`, never `train.log`.

| block | s/gen | mean plies | gen0 |
|---|---|---|---|
| compile-on-1 | 93.7 | 126.0 | 88 s |
| compile-off-1 | 127.7 | 127.7 | 121 s |
| compile-on-2 | 100.2 | 126.0 | 83 s |
| compile-off-2 | 136.2 | 127.7 | 125 s |

**compile ON 97.0 s/gen, OFF 131.9 s/gen — 1.36x.**

Against the 1.48x total gap, compile accounts for **about 75%**. The remaining
quarter is the newer torch and the bridge work, which this test does not separate.

## Why the blocks were alternated, and why it mattered

Both arms were slower in the second block than the first: compile-on 93.7 → 100.2,
compile-off 127.7 → 136.2. That is real drift over the hour — roughly 7% on each
arm — from a machine that warms up and from background load.

Run A-then-B instead of A-B-A-B and all of that drift lands on whichever arm went
second. Had compile-off run second as a single block, the ratio would have come
out near 1.45x instead of 1.36x: a 7% overstatement produced entirely by the
schedule. Measured within blocks, the two pairs agree exactly — 1.36x and 1.36x.

## Conditions

The machine was quiet: `runs/long-oct` had finished, no FEA (srm-studio-34 held
it for the hour), load 1.93 at start. The OpenClaw gateway that had been
crash-looping at ~130% CPU for two weeks was uninstalled the day before, so
unlike every measurement before 2026-10-25 this one did not run alongside it.

A throughput measurement taken under contention measures the contention, which
is why the script refuses to start while `chess-train` is alive.

## What this does not say

It does not say the speedup is 1.36x end to end — it says compile is 1.36x of a
1.48x whole. The other quarter is unattributed, and splitting torch-version from
bridge work would need a second A/B against a pinned older torch.
