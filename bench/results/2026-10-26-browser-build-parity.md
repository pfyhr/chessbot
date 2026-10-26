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

---

## Addendum: three bugs the first gate could not see

Same day, from real use of the deployed page. All three were invisible to a
gate that only checked numbers.

### The backend was wrong, and said so in a footnote

Safari picked WebGPU and the page's own load-time self-check fired at 0.42 —
printed as "this backend may be computing something slightly different", which
badly undersold it. Measuring it properly, against ONNX Runtime on CPU with the
same fp16 model over the same 127 positions:

| | worst logit | worst value | top-1 agree | mean KL | search agrees |
|---|---|---|---|---|---|
| wasm CPU | reference | reference | 127/127 | — | 12/12 |
| WebGPU (Safari) | 0.724 | 0.143 | 122/127 | 1.2e-03 | 11/12 |

A 0.14 error on the value head is not rounding, and it changed the move a
fixed-seed search played in 1 of 12 cases. Meanwhile the Node and Chromium
gates passed, because neither runs WebGPU.

Two things came out of this. The page now **verifies whatever backend it
starts** against reference vectors before using it, falls back to one that
passes, and says plainly that the moves are not the network's if nothing does.
And the GPU became opt-in (`?backend=webgpu`) rather than preferred — because
it is also **slower**: 68 evaluations/second against 205 for the wasm kernels
on the same machine. This search evaluates one leaf at a time, so a batch of
one pays full dispatch overhead for 75M multiply-adds. The GPU was losing on
both axes.

### The board was not square, in one engine only

`aspect-ratio: 1` made the *container* square while the eight rows inside were
implicit and `auto`-sized — to the height of a piece glyph, not to an eighth of
the board. WebKit renders that as visibly collapsed empty ranks. Chromium
stretches the rows and looked perfectly fine; with the fix reverted, headless
Chromium still reported square squares and was caught only by a 2px shortfall
in tiling. **A single-engine gate would have passed the broken build.**

Compounding it, the glyph was sized `min(8.4vw, 3.3rem)` — against the
*viewport*, while the board is one column of a two-column grid. At 1080px the
board is ~700px wide, so a square is ~87px and the glyph asked for 90px: the
piece was wider than the square it sat in. Now sized in `cqmin` against the
square, with `container-type: size` so content can never feed back into the
track at all.

### The result was computed and then hidden

Two games ended in mate and the page said nothing. It had detected both
correctly — the board was non-interactive, the engine did not reply — but
`#status` lived inside the `#loading` div, which is hidden once the network
finishes downloading. Every message after boot, the result included, was
written into an element nobody could see.

There is no clever lesson here beyond the obvious one: *all three of these were
visible in two seconds of looking at the page, and none of them were visible to
a gate that checked 127 positions to four decimal places.* The numerical gate
was not wrong, it was answering a different question. The layout gate now runs
the real page in both engines, and replays both mates to assert the result is
shown, no further moves are accepted, and the evaluation panel is not still
listing candidates from a position that is two moves old.
