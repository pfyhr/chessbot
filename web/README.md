# The browser build

`gen2600` playable in a tab, with no server and nothing sent anywhere. The
network, the search and the move generator all run client-side.

Live: **https://pfyhr.github.io/chessbot/**

## What actually runs

| piece | where it comes from | size on the wire |
|---|---|---|
| Gumbel MCTS, position encoding, move generation | `crates/az-wasm` → `az_wasm.wasm` | 0.1 MB |
| the network | `runs/long-oct/gen2600.pt` → ONNX fp16 | 19.0 MB |
| ONNX Runtime Web (WebGPU + wasm kernels) | npm, vendored at build time | 5.7 MB |
| the six piece glyphs | Noto Sans Symbols 2, subset at build time | 1.7 KB |

About 25 MB on a first visit, cached after that.

The important thing is the first row. The page does **not** reimplement the
search or the encoding in JavaScript — it compiles the same Rust the trainer
uses to `wasm32-unknown-unknown` and calls into it. `az-core`'s MCTS was
already written as a resumable state machine, because self-play pools leaf
evaluations from hundreds of games into one network batch, and that is exactly
the shape an asynchronous evaluator needs:

```js
eng.search(async (obs) => net.evaluate(obs), { sims: 128, considered: 32 });
```

There is no `wasm-bindgen`. The surface is a position, a search and a scratch
buffer for strings, which is small enough that hand-written glue is less
machinery than a code generator — and it keeps CI to `cargo build --target
wasm32-unknown-unknown` with no extra toolchain to pin. The module imports
nothing and is 879 KB.

## The parity gate

A browser port of a search and an encoding fails *silently*. A transposed plane
still trains, still plays, and is simply a different network; a policy index off
by one plane maps every move to its neighbour's logit. Nothing raises. So the
build does not ship until it reproduces the trainer:

```bash
./build.sh                  # export, assemble
node tools/parity_check.mjs # the gate
node tools/smoke.mjs        # the shipped module plays a short game
```

Two kinds of claim, kept apart deliberately:

* **Exact** — FEN, legal move list *and order*, policy indices, and a hash of
  the raw observation bytes, over 127 positions chosen to include promotions,
  en passant, castling and threefold repetition. Same Rust, different target. A
  tolerance here would hide the bug it exists to catch.
* **Within tolerance** — network outputs, because ONNX Runtime is not PyTorch
  and fp16 is not fp32. Judged on *decisions*: the top legal move, and the move
  a fixed-seed search actually plays.

Current result: all 127 exact, 12/12 fixed-seed searches choose the same move
with root values agreeing to four decimals.

`tools/parity_ref.py` writes the reference straight out of the training code;
`web/parity/reference.json` is committed so CI can run the gate without PyTorch
or the 131 MB checkpoint.

### Why fp16

Measured, not assumed. Against the fp32 export over the same 127 positions:

| | max logit Δ | max value Δ | top-1 agree | mean KL |
|---|---|---|---|---|
| ONNX fp32 | 0.0025 | 0.00009 | 127/127 | 1.3e-08 |
| ONNX fp16 | 0.298 | 0.011 | 125/127 | 1.1e-05 |

The two disagreements are ties the network scores equal to four decimal places
— logit gaps of 0.0012 and 0.0002. At 128 simulations, fp16 and fp32 chose the
same move in all 12 search cases. It halves a 44 MB download for no decision
the network itself can tell apart.

### What the gate caught

Vendoring ONNX Runtime looks like copying two files -- the bundle and the wasm
binary -- and that is wrong. `ort.bundle.min.mjs` *dynamically imports* the
`.jsep.mjs` glue, which then fetches the binary; the bundle does not inline it,
despite the name. Ship two of the three and you get `no available backend
found` at runtime, with the board, the search and every other part of the page
working perfectly.

Both gates therefore import ORT from `dist/vendor`, not from `node_modules`.
Testing the npm package would have tested a dependency that was correctly
installed rather than the copy that was incorrectly made.

### The layout and terminal-state gate

Geometry needs a layout engine, so `tools/layout_check.mjs` runs the built site
in real browsers -- **both Chromium and WebKit**, at phone, tablet and desktop
widths -- and asserts the board is square, every square is square, eight of
them tile the content box, nothing overflows horizontally, and the page reaches
a playable state with no self-check warning.

Two engines because one was not enough. The board had `aspect-ratio: 1`, so the
*container* was square, but its eight rows were implicit and sized `auto` -- to
the height of a piece glyph rather than an eighth of the board. WebKit renders
that as visibly collapsed empty ranks; Chromium stretches the rows and looked
fine. A Chromium-only gate would have passed the broken build.

It also replays two real games that ended in mate and asserts the page says
`White wins by checkmate`, accepts no further moves, does not answer, and is
not still showing the previous search's candidates. Games are loaded through
`?moves=`, a real feature rather than a test hook, so the test takes the path a
visitor takes.

### The promotion picker, and why pixels

Four choices used to be a 2x2 grid inside the single promotion square, making
each glyph a quarter of a square. At that size a fixed 1.3px centred outline is
~6% of the glyph and eats the fill, so **white** pieces rendered as dark blobs
on a dark panel. Every computed style was correct the whole time -- the right
colour, the right font size -- which is why the gate reads pixels: it decodes
Playwright's screenshots with `node:zlib` and asserts the piece's own tone
dominates its cell.

That gate then caught something a single engine would not have: WebKit draws
`-webkit-text-stroke` far heavier than Chromium. The same declaration gave 0.5%
dark pixels in one and 4.0% in the other, and the queen -- the thinnest-stroked
glyph -- tipped over. Both colours now use an eight-direction shadow ring sized
in `em`, which sits outside the glyph instead of eating into it and renders the
same in both engines.

The rule that came out of it: **outline widths in `em`, never `px`.** A width
that is 2% of an 88px board glyph is 6% of a 22px one.

### The piece font, and why it is not the platform's

The board used to write `U+2654..265F` and let the platform choose a font. On
iOS every pawn came out as a glossy black emoji, in both colours, so White had
eight black pawns. `U+265F` is the only chess symbol in Unicode carrying an
Emoji property, which is why the pawns alone broke; and on the same device the
black queen and king rendered as hollow outlines, indistinguishable from the
white ones at a glance.

The conventional fix is `U+FE0E`, the text-presentation variation selector. It
was measured before it was trusted, and rejected: in WebKit it also changes
which font is selected, costing 27% of advance width and leaving the pieces
noticeably small. So the glyphs ship instead — six of them, subset out of Noto
Sans Symbols 2 at 1,672 bytes, regenerated by `tools/make_piece_font.sh`.

Vertical centring lives in the font's `hhea`/`OS/2` metrics, derived from the
glyph outlines by that script, not nudged in CSS. The `@font-face` deliberately
has **no `unicode-range`**: a ranged face is excluded from the strut, so its
ascent and descent are never consulted and the centring is silently inert.

**`chess-pieces.OFL.txt` must stay beside the font, in `web/src/` and in
everything published.** The OFL requires the copyright notice and licence to be
distributed with the font in original or modified form, and this one is
modified — subset, with rewritten metrics. `build.sh` copies both files into
`dist/src/`; shipping the `.woff2` without the licence would put the deployed
site out of compliance. If the font is ever replaced, the replacement's licence
takes its place rather than the line being deleted.

Possible later: **drawing our own pieces.** Six SVG paths would drop the
licence question entirely, let the knight read at 22px in the promotion picker
where Noto's is busiest, and give the board a look of its own instead of a
system font's. It is not a bug and nothing is waiting on it — the font renders
correctly in both engines and on-device. It is a want, recorded so it is not
rediscovered.

### Why the GPU is opt-in

`?backend=webgpu` asks for it; the default is the wasm kernels. That is the
opposite of what you would expect, and it is what the measurements say.

Against ONNX Runtime on CPU with the same fp16 model, Safari's WebGPU path
came out at:

| | worst logit | worst value | top-1 agree | mean KL | search agrees |
|---|---|---|---|---|---|
| wasm CPU | reference | reference | 127/127 | — | 12/12 |
| WebGPU | 0.724 | 0.143 | 122/127 | 1.2e-03 | 11/12 |

A value-head error of 0.14 is not rounding, and it changed the move a
fixed-seed search played. It is also **slower**: 68 evaluations/second against
205 for wasm on the same machine, because this search evaluates one leaf at a
time and a batch of one pays full GPU dispatch overhead for 75M multiply-adds.

So the page verifies whichever backend it starts against reference vectors
before using it, falls back to one that passes, and says plainly in the UI if
nothing does. A backend that is subtly wrong is worse than one that fails,
because nothing reports it.

### Why Node, not a headless browser

ONNX Runtime Web's wasm backend is the same code the page falls back to, so
this gate tests a path that really ships. A headless-Chrome WebGPU run would
cover the *other* path but needs a GPU in CI and fails for reasons unrelated to
the model. WebGPU is covered instead by the page's own load-time self-check,
which evaluates the start position, compares it against these same numbers, and
says so in the UI if the backend disagrees.

## Speed

205 evaluations/second on single-threaded wasm CPU (M-series laptop), so a
128-simulation move takes about 0.6 s with no GPU at all. WebGPU is used where
available. GitHub Pages cannot send COOP/COEP headers, so there is no
`SharedArrayBuffer` and therefore no wasm threads — `numThreads` is pinned to 1
rather than left to discover that on every load.

## Building

```bash
cd web
npm install
./build.sh                       # needs .venv with torch + onnx, and the checkpoint
node tools/parity_check.mjs
python -m http.server -d dist 8099
```

`SKIP_MODEL=1` reuses an already-exported model in `dist/model` instead of
running the exporter — that is the path CI takes, since a runner has neither
PyTorch nor the checkpoint.

## Deploying

`.github/workflows/pages.yml` builds the wasm, downloads the exported network
from the `weights-gen2600` release, runs both gates, and publishes to Pages.
The deploy is downstream of the gate: parity failing fails the job.

The weights are a release asset rather than a file in the repo — 22 MB in git
history is 22 MB forever, and this way the published network has a version and
notes of its own.

## Honest limits

* It is a 10.99M-parameter network that trained at roughly 1.6 plies of search.
  It gained +145 Elo over its last 72-hour run and it is still not a strong
  engine. It will hang pieces.
* No clock, no opening book, no endgame tables. That is the point.
* Threefold repetition is detected from the moves played in the tab. A position
  loaded from FEN starts with no history.
