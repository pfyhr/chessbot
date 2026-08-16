# chessbot

An AlphaZero-style self-improving chess engine. Rust core, PyTorch training, built to run
on a single Apple Silicon machine.

Three goals, in order:

1. A **demonstrably self-improving** RL loop — Elo rises over generations, verified by SPRT.
2. A **fast** loop on an M2 Max (no CUDA). Performance engineering is a feature, not a
   postscript.
3. Learning Rust by writing the hot path in it.

## Status

- [x] **Phase 0** — workspace scaffold, toolchain
- [x] **Phase 1a** — perft correctness + movegen backend benchmark
- [x] **Phase 1b** — position/move encoding (4672-move policy), cross-checked vs `python-chess`
- [x] **Phase 1c** — PyO3/maturin bridge, measured against a network forward
- [ ] Phase 2 — Gumbel MCTS + self-play, validated on Connect4
- [ ] Phase 3 — chess self-play + SE-ResNet training
- [ ] Phase 4 — UCI binary, time-ladder checkpoints, fastchess/Ordo harness
- [ ] Phase 5 — performance engineering
- [ ] Phase 6 — tabula-rasa vs warm-start experiment

## Layout

```
crates/az-core     game abstraction, encodings, Gumbel MCTS, self-play driver
crates/az-perft    perft correctness + movegen throughput benchmark
tests/             cross-language checks against python-chess
bench/results      committed benchmark results and the decisions they drove
```

## Encoding

Policy is AlphaZero's 4672 = 73 planes x 64 origin squares, flat index
`plane * 64 + from_square` — plane-major, matching a `(73, 8, 8)` conv head flattened in
PyTorch's `(C, H, W)` order. Observations are 119 planes (8 history steps x 14, plus 7
metadata). Everything is from the mover's perspective; Black's board is mirrored vertically.

Verified two ways, because they catch different things:

- **Rust unit tests** — injectivity and round-trip over a 200-game random walk. These prove
  the encoding is *a* bijection.
- **[`tests/test_encoding_vs_python_chess.py`](tests/test_encoding_vs_python_chess.py)** — a
  second encoder written from the paper's description, checked against the Rust one over
  **61,547 positions / 1,815,170 moves**. This is what proves it is *AlphaZero's* bijection:
  a coherently wrong plane layout passes every self-consistency test while training the
  network against targets that mean nothing.

```
cargo build --release --bin dump
python3 -m pytest tests/ -v
```

## Node cost

`Game::expand` returns the legal moves and the terminal outcome from a single move
generation. Asking separately costs three movegens per node, because shakmaty's
`is_checkmate` and `is_stalemate` each regenerate the list internally — measured at
**1.9x** on the hottest operation in the search:

```
./target/release/nodecost
```

It verifies both spellings agree on all 4000 sampled positions before reporting a
timing, so the fast path cannot quietly become the wrong path.

## Bridge cost

```
.venv/bin/python bench/bridge_throughput.py
```

The boundary costs **0.0–2.5%** of encode-plus-forward — it is not the bottleneck, and the
once-per-batch design has room to spare. See
[`bench/results/2026-08-16-bridge-throughput.md`](bench/results/2026-08-16-bridge-throughput.md).

The number that shapes Phase 2 is the other one: MPS has a ~2 ms fixed cost per call, so a
batch of 1 gets 482 positions/sec against 24,411 at batch 512 — **50x from batching alone**.
Self-play has to run ~128 games concurrently and pool their leaf evaluations, or the GPU
sits 98% idle.

## Design notes

**Inverted control flow.** Python drives the training loop, but Rust owns every board and
search tree. The FFI boundary is crossed once per neural-net batch, never per node:

```python
gen = chessbot_core.SelfPlay(games=256, sims=32)
while not gen.done():
    obs = gen.next_batch()        # Rust steps all games until N leaves need eval
    p, wdl, mlh = net(obs)        # PyTorch on MPS
    gen.submit(p, wdl, mlh)       # Rust expands, applies virtual loss, continues
```

**Gumbel AlphaZero**, not vanilla PUCT. Root Gumbel-top-k plus sequential halving, with the
completed-Q improved policy as the training target. It keeps learning at very low simulation
budgets, so we run 32 sims instead of 800 — roughly a 25x throughput multiplier, and the
difference between this working on one machine and not.

**Generic over a `Game` trait**, with Connect4 alongside Chess. Connect4 converges to
near-perfect play in an afternoon, giving a hard pass/fail on the RL loop before chess —
where a subtly wrong training target is indistinguishable from slow learning for weeks.

## Building

Requires a Rust toolchain (`rustup`) and Python 3.11+.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/maturin develop --release     # builds the Rust core into the venv

cargo build --release
cargo test --release
.venv/bin/python -m pytest tests/ -v
```

Always build in release. Debug builds are 20–50x slower and any timing from one is
meaningless — which is why `[tool.maturin] profile = "release"` is set even for
`maturin develop`.

The `extension-module` feature is off by default and enabled only by maturin. With it
always on, `cargo test` builds a standalone binary whose Python symbols resolve to nothing
and it aborts at startup.

## Perft

```
./target/release/az-perft verify --max-depth 4      # correctness gate, 48 reference counts
./target/release/az-perft bench --depth 5 --repeat 5
```

Movegen backend is **shakmaty**, chosen on measurement — see
[`bench/results/2026-08-15-movegen-backend.md`](bench/results/2026-08-15-movegen-backend.md).
The short version: cozy-chess is up to 2.6x faster at bulk move *enumeration*, but shakmaty
is 3–22% faster on the make-move path, and make-move is what MCTS actually does.

`perft(6)` from startpos: 119,060,324 nodes in 1.90s, single-threaded.
