# The browser build, and what it took to trust it

2026-10-26. `runs/long-oct/gen2600` now plays in a browser tab with no server:
<https://pfyhr.github.io/chessbot/>. The interesting part is not the page, it is
the gate in front of it.

## The failure mode this is guarding against

Porting a search and a position encoding to a second platform is the kind of
change that cannot fail loudly. A transposed observation plane still trains,
still plays legal chess, and is simply a *different network*. A policy index off
by one plane maps every move to its neighbour's logit. Neither raises an
exception; both produce a page that looks like it works and plays like the
network was never trained.

So the design rule was: **do not reimplement anything.** `crates/az-wasm`
compiles the same `az-core` the trainer uses to `wasm32-unknown-unknown` and
exposes a flat C ABI. The page's move generator is shakmaty, its encoding is
`chess.rs`, its search is `mcts.rs`. There is no second chess implementation to
disagree with the first.

That worked out better than expected because `az-core`'s MCTS was *already*
resumable — `prepare` / `pending` / `apply` / `result` — since self-play pools
leaf evaluations from hundreds of games into one network batch. An asynchronous
JavaScript evaluator wants exactly that shape. The browser runs the trainer's
own loop with a batch of one.

No `wasm-bindgen`: the surface is small enough that hand-written glue is less
machinery than a code generator, and the module ends up importing nothing at
all. 879 KB, 7-second build.

## The gate

`web/tools/parity_check.mjs`, run on every deploy. 127 positions, chosen to
include promotion, en passant, castling and threefold repetition, reached by
move sequences so the history planes are exercised rather than zeroed.

**Exact** — FEN, legal move list *and order*, policy index of every legal move,
and an FNV-1a hash of the raw observation bytes. Same Rust, different target, so
a tolerance here would only hide the bug it is meant to catch.

> all 127 exact

**Within tolerance** — network outputs, judged on decisions rather than floats.

| | max logit Δ | max value Δ | top-1 agree | mean KL | max KL |
|---|---|---|---|---|---|
| ONNX fp32 vs torch | 0.0025 | 0.00009 | 127/127 | 1.3e-08 | 2.2e-07 |
| ONNX fp16 vs torch | 0.298 | 0.011 | 125/127 | 1.1e-05 | 1.4e-04 |

**Whole stack** — fixed-seed search, wasm MCTS against native MCTS on the same
model. 12/12 the same move, root values agreeing to four decimal places.

## fp16 was a measurement, not a guess

fp16 halves a 44 MB download, and the first look at it is alarming: it moves the
top policy move in 2 of 127 positions. Both turn out to be ties.

```
case  33  logit gap 0.00120   c5d4 -> c6d4   p=0.1781 vs 0.1779
case  85  logit gap 0.00021   h3h4 -> f2f4   p=0.1139 vs 0.1139
```

The network scores those pairs equal to four decimal places. Picking either is
the same decision in any sense that matters. And at 128 simulations with the
same seed, fp32 and fp16 chose the **same move in all 12** search cases — search
averages over many evaluations and is far less sensitive than a single argmax.

Shipping fp16. This is the same lesson as the `vs_prev` misreading three weeks
ago in the other direction: a proxy that looks bad (2/127 disagreements) was
worth checking against the decision that actually matters (0/12 moves changed)
before acting on it.

## Why the gate runs in Node rather than a browser

ONNX Runtime Web's wasm backend is the same code the page falls back to when
WebGPU is unavailable, so a Node run tests a path that genuinely ships. A
headless-Chrome WebGPU run would cover the other path, but it needs a GPU in CI
and fails for reasons that have nothing to do with the model — a gate that
fails for unrelated reasons stops being a gate, because people start ignoring
it.

WebGPU is covered differently: the page evaluates the start position on load and
compares it against the same reference numbers, measured against the *shipped*
model under ONNX Runtime on CPU rather than against torch. Comparing to torch
would need a tolerance wide enough to swallow fp16 rounding, and a tolerance
that wide would not notice a WebGPU kernel quietly computing the wrong thing.
If the backend disagrees, the page says so in the UI instead of playing on.

## Numbers

| | |
|---|---|
| wasm core | 879 KB, imports nothing |
| network | 22.1 MB fp16 (19.0 MB gzipped), 10.99M params |
| ONNX Runtime | 23.7 MB (5.6 MB gzipped), WebGPU + CPU kernels in one binary |
| first visit | ~25 MB over the wire, cached after |
| speed | 205 evals/s single-threaded wasm CPU → ~0.6 s per 128-sim move |

GitHub Pages cannot set COOP/COEP, so there is no `SharedArrayBuffer` and no
wasm threads regardless of what is requested. `numThreads` is pinned to 1 so
ORT does not probe for workers it will never get.

## What this does not claim

The page is a distribution channel, not an evaluation. It says nothing about
strength that `2026-10-26-long-oct-the-probes-were-wrong.md` did not already
say: +144.92 ± 50.73 Elo over the run's starting checkpoint at 128 nodes, and
the network will still hang pieces.
