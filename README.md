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
- [x] **Phase 2** — Gumbel MCTS + batched self-play, validated on Connect4
- [ ] Phase 3 — chess self-play + SE-ResNet training
- [ ] Phase 4 — UCI binary, time-ladder checkpoints, fastchess/Ordo harness
- [ ] Phase 5 — performance engineering
- [ ] Phase 6 — tabula-rasa vs warm-start experiment

## Layout

```
crates/az-core     game abstraction, encodings, Gumbel MCTS, self-play driver
crates/az-perft    perft correctness + movegen throughput benchmark
crates/az-py       PyO3 bindings -> the `chessbot_core` Python module
python/chessbot    network, training loop, evaluation gates
tests/             cross-language checks against python-chess
bench/results      committed benchmark results and the decisions they drove
```

## Playing against a generation

Every generation is a saved checkpoint, so "how good was it after N generations?" is a
question you answer by playing it:

```
PYTHONPATH=python .venv/bin/python -m chessbot.play_connect4 --gen 39
PYTHONPATH=python .venv/bin/python -m chessbot.play_connect4 --ladder
```

`--ladder` runs a staircase: win and you face a later generation, lose and you drop back,
so it finds your level in a handful of games rather than forty. This is a rehearsal for the
chess time-ladder in Phase 4, where checkpoints are saved against wall-clock instead of
generation — so the question becomes *how many minutes of training* it takes to pass you.

## Running the Connect4 loop

Connect4 is not a side quest — it is the gate. A correct RL loop converges here in
minutes, so a wrong one is *loud* instead of looking like "chess is slow to learn".

```
PYTHONPATH=python .venv/bin/python -m chessbot.train_connect4 --generations 40
```

Four gates, ordered by how much they tell you:

| gate | what it proves |
|---|---|
| **tactics** | positions with a forced win or forced block, generated from real play and checked against exact ground truth — cannot be gamed by a degenerate policy |
| **centre** | Connect4 is solved and the first player wins only by taking the middle; nothing in the loop is told this |
| **vs-random** | the floor — anything that learned at all clears it |
| **vs-prev** | is generation N actually better than N-1 |

Match play randomises openings. Greedy policy play is deterministic, so without that
a 200-game match is two distinct games repeated 100 times, and the score it reports is
noise dressed as data.

### Result

40 generations, 10,240 games, **14 minutes** on an M2 Max
([full report](bench/results/2026-08-19-connect4-loop.md)):

| metric | gen 0 | gen 39 |
|---|---:|---:|
| tactics, raw policy | 0.295 | **0.583** |
| tactics, with search @32 | 0.605 | **0.750** |
| centre preference | 0.193 | **0.999** |
| vs random | 0.718 | **0.975** |

Last against first over 400 games: **395W–5L on raw policy** (+759 Elo), 357W–43L
with search (+368 Elo). The searched margin is narrower because search partly
compensates for a weak policy — that is the honest number for how much better the
*engine* got, while the raw margin is how much better the *network* got.

The centre figure is the one worth trusting most: Connect4 is solved and the first
player wins only by taking the middle, and nothing in the loop is told that.

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
sp = chessbot_core.Connect4SelfPlay(concurrency=256, total_games=256, sims=32)
while (obs := sp.next_batch()) is not None:   # Rust steps every game to its next leaf
    logits, wdl = net(obs)                    # PyTorch on MPS, one forward
    sp.submit(logits, wdl_to_scalar(wdl))     # Rust expands, backs up, advances
obs, policy, z = sp.take_training_data()
```

**One leaf per tree, many trees.** Because Gumbel runs at a small simulation budget, each
tree only needs one leaf evaluated per pass — so concurrency comes from the number of games,
not from forcing one tree to yield several leaves. That removes virtual loss from the design
entirely. Batch size is simply the live game count.

**Gumbel AlphaZero**, not vanilla PUCT. Root Gumbel-top-k plus sequential halving, with the
completed-Q improved policy as the training target. It keeps learning at very low simulation
budgets, so we run 32 sims instead of 800 — roughly a 25x throughput multiplier, and the
difference between this working on one machine and not.

The sigma transform has two details that are easy to miss and fatal to skip: completed-Q
values are **min-max rescaled to [0, 1]** and scaled by **0.1**, not 1.0. With raw Q in
[-1, 1] and a unit scale the term reaches ~60 against O(1) logits, so the improved policy
collapses onto argmax-Q and discards the network's prior. The symptom is subtle — the loop
still trains, just badly.

**Generic over a `Game` trait**, with Connect4 alongside Chess. Connect4 converges in
minutes, giving a hard pass/fail on the RL loop before chess — where a subtly wrong training
target is indistinguishable from slow learning for weeks.

## Building

Requires a Rust toolchain (`rustup`) and Python 3.11+.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/maturin develop --release     # builds the Rust core into the venv

cargo build --release
cargo test --release
.venv/bin/python -m pytest tests/ -v          # 46 tests, no PYTHONPATH needed
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
