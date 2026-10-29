# Handoff — 2026-10-28

Nothing is running. Jupiter is free. Working tree clean, `main` pushed, the
Pages site is live and green.

## Done since the last handoff

**The browser build.** `gen2600` plays in a tab at
<https://pfyhr.github.io/chessbot/>, client-side, no server. It is not a
reimplementation — `crates/az-wasm` compiles the same `az-core` the trainer
uses, and the deploy is gated on reproducing the trainer exactly (FEN, legal
move order, policy indices, observation hash, fixed-seed searches).
See [`web/README.md`](../web/README.md) and
[`2026-10-26-browser-build-parity.md`](../bench/results/2026-10-26-browser-build-parity.md).

**The piece font.** The page used to borrow `U+2654..265F` from the platform.
On iOS every pawn rendered as a glossy black emoji — `U+265F` is the only chess
symbol with a Unicode Emoji property — and the black queen and king came out as
hollow outlines. Fixed by shipping a 1.7 KB subset of Noto Sans Symbols 2
(`web/src/chess-pieces.woff2`, regenerate with `web/tools/make_piece_font.sh`).
The conventional `U+FE0E` fix was measured first and rejected: it costs 27% of
advance width in WebKit. Vertical centring is baked into the font's
`hhea`/`OS/2` metrics, derived from the glyph outlines, not nudged in CSS.

**Three gates in `web/tools/layout_check.mjs`**, running in Chromium and WebKit
on every deploy: the font actually loaded; each piece reads light or dark by a
signed score against itself in the other colour; the ink is centred in its
square to within 4%. Each was validated by injecting the bug it is meant to
catch. CI last read centring −1.4% / +2.2%.

**Driver threading: measured, and it does not pay.** Game-granularity threading
is implemented and correct (slots sharded across K threads, each owning whole
games, reproducible ply-for-ply at any K). It bought **+0.5% ± 9.8%** on a real
generation; driver s/gen went 6.4 → 5.9 → 6.1 at 1/2/4 threads — flat. Isolated
ceiling 1.27×, and 8 threads is worse than 4. Default is `threads=1`.
[`2026-10-26-driver-threads.md`](../bench/results/2026-10-26-driver-threads.md).

**The 97.0 s baseline was wrong.** It included a ~12 s probe suite
(`evaluated: true` on all 2,602 long-oct generations). Corrected to **85.2
s/gen, 42.3 gens/h**, and every derived threshold moved in
`bench/vast_deploy.sh` and `docs/rented-training-plan.md`. It had been close to
flipping a rental verdict.

## Open

**The rented-GPU measurement session** — the next real piece of work, and it is
blocked on Pontus, not on code. `docs/rented-training-plan.md` §5:

1. Fund a Vast.ai account; $20 covers all three GPU classes.
2. Pick an offer — sort by **CPU single-thread quality**, not core count or
   DLPerf, because the driver is 53–92% of every row. Set **disk ≥ 100 GB at
   rent time**; it cannot be changed afterwards.
3. `./bench/vast_deploy.sh <host> <port> check` first (installs nothing, rules
   out a bad box in a minute), then `measure`.
4. Decide against the written thresholds in §1, not against effort spent.

Two questions, not one: *is a rented card worth it* (the 4090 straddles the
bar), and *are tensor cores reachable at this net size* (TF32 is worth 7.4× on
an H100 and nothing on a GeForce card, and PyTorch enables it by itself).

**Loose ends**

* `/Users/pontus/repos/chessbot` is hardcoded in 13 files (`bench/*.sh`,
  `config.json`) and in one old commit of `web/parity/reference.json`. **Closed
  as won't-fix on 2026-10-29** — Pontus looked and did not care, and nothing is
  secret in a home directory path. Do not re-raise it as a leak. The `R=` lines
  are still a real portability bug the day these scripts run on a rented box;
  fix them then, as `R=$(cd "$(dirname "$0")/.." && pwd)` the way `web/build.sh`
  already does.
* `?backend=webgpu` is untested surface: CI runners have no GPU.
* Wanted, not needed: **our own piece glyphs**, six SVG paths instead of a
  subset of Noto Sans Symbols 2. Would drop the font licence from the deploy
  and read better at promotion-picker size. Nothing waits on it.
* Still open and cheap, from `STATUS.md`: `c_scale` is 0.1 where the paper uses
  1.0, and `chess_eval.py:213` hardcodes `max_considered=16`.

## Commands to resume

```bash
# play
.venv/bin/chess-play --run runs/long-oct --gen 2600   # or chess-serve / chess-uci
.venv/bin/c4-play --ladder                 # the fast sanity loop

# the browser build
cd web && npm install && ./build.sh        # needs .venv + the checkpoint
node tools/parity_check.mjs                # the gate; exits non-zero on drift
node tools/smoke.mjs
node tools/layout_check.mjs
python -m http.server -d dist 8099
# SKIP_MODEL=1 reuses dist/model — the path CI takes

# bench
./bench/driver_threads.sh                  # needs the machine to itself
./bench/vast_deploy.sh <host> <port> check | measure

# deploy
git push origin main                       # web/** or crates/** triggers pages.yml
gh workflow run pages.yml                  # or by hand
gh run watch
```

## What a fresh session should know

* **Benchmarks need the machine to themselves.** A contended match read
  +137 ±49 where an idle one read +241 ±73 on the same pair. Check mtimes and
  `ps` before believing any A/B.
* **Driver seconds per generation is the number that transfers.** Wall clock is
  not; it moves with whatever else is on the box.
* **A recorded baseline may be measuring different work.** This bit twice now.
  Re-measure the control in the same session as the arm.
* **Before believing "no change", feed the measurement a case it must fail.**
  The font centring read pixel-identical because the fix was inert *and* the
  instrument was blind, at the same time. A gate that has never failed on
  purpose is not known to work.
* **SPRT decides, fixed length publishes.** Its point estimate is biased upward
  — +249 became +145 on the same pair.
* Pushing anything under `.github/workflows/` needs the `workflow` token scope.
* Claude attribution is wanted on this repo; the public attribution is
  deliberate.
