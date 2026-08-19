# Phase 2 gate: does the reinforcement-learning loop learn?

**Date:** 2026-08-19
**Machine:** Apple M2 Max, PyTorch 2.13 on MPS
**Run:** 40 generations, 256 self-play games each, 32 simulations per move
**Net:** 4x64 SE-ResNet, 496k parameters
**Cost:** 10,240 games, 6,258,147 network evaluations, **14.1 minutes**

```
PYTHONPATH=python .venv/bin/python -m chessbot.train_connect4 \
    --generations 40 --games 256 --concurrency 256 --sims 32 \
    --steps 200 --window 8 --out runs/connect4-v1
PYTHONPATH=python .venv/bin/python -m chessbot.connect4_report runs/connect4-v1
```

## Result

| metric | generation 0 | generation 39 |
|---|---:|---:|
| tactics, raw policy | 0.295 | **0.583** |
| tactics, with search @32 | 0.605 | **0.750** |
| centre-column preference | 0.193 | **0.999** |
| score vs random | 0.718 | **0.975** |

Head to head, 400 games with randomised openings:

| condition | result | score | implied Elo |
|---|---|---:|---:|
| raw policy | 395W 5L 0D | 0.988 | **+759** |
| search @32 | 357W 43L 0D | 0.892 | **+368** |

## Reading it

**The centre metric is the one that matters most.** Connect4 is solved: the first
player wins only by taking the middle column. Nothing in the training loop is told
this — there is no opening book, no heuristic, no shaped reward. The network went
from 0.193 (below the 1/7 you would get by chance) to 0.999 on its own. That is
independent evidence the loop learns real structure rather than merely reducing a
loss.

**Search and policy improve together, and the gap between them is informative.**
The raw-policy head-to-head shows a much larger gap (+759 Elo) than the searched
one (+368). That is expected and reassuring: search partially compensates for a
weak policy, so generation 0 *with* search is far stronger than generation 0
without. The narrower searched margin is the honest number for "how much better is
the engine", and the wider raw margin is the honest number for "how much better is
the network".

**`vs-prev` sitting near 0.5 is not a failure.** It compares neighbouring
generations, and once a run converges neighbours are close by construction. It is
the wrong statistic for "did this work" — hence the first-versus-last comparison
above.

## What is not solved

Tactical accuracy tops out at 0.583 raw / 0.750 with search, not 1.0. The suite is
built from real play and verified against exact ground truth, so these are genuine
misses on positions with a forced win or a forced block. A 496k-parameter net at 32
simulations does not fully master Connect4 tactics in 14 minutes, and it would be
wrong to present it as though it had. More simulations, more generations, or a
larger net would all move this; none were tried, because the gate is "the loop
learns", not "Connect4 is solved".

## The bug this run found

The first attempt at this run had `centre` *falling* generation over generation and
`vs-prev` pinned at 0.5. The cause was in the sigma transform: completed-Q values
were being used raw, in [-1, 1], with `c_scale = 1.0`, so the term reached ~60
against logits of order 1. The improved policy became `softmax(~60 * q)` — a near
one-hot on argmax-Q at a simulation budget where Q is mostly noise, discarding the
network's prior entirely.

DeepMind's `mctx` min-max rescales completed-Q to [0, 1] across the node and uses
`value_scale = 0.1`. Neither detail is in the paper's main text, and the loop still
*trains* without them, just badly. Same seed and hyperparameters, generation 1:

```
before: tactics 0.448  centre 0.083  vs-prev 0.515
after:  tactics 0.347  centre 0.399  vs-prev 0.740
```

This is precisely the failure mode Connect4 exists to catch. On chess it would have
looked like "learning is slow" for weeks.
