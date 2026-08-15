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
- [ ] Phase 1b — position/move encoding (4672-move policy), bijection tests vs `python-chess`
- [ ] Phase 2 — Gumbel MCTS + self-play, validated on Connect4
- [ ] Phase 3 — chess self-play + SE-ResNet training
- [ ] Phase 4 — UCI binary, time-ladder checkpoints, fastchess/Ordo harness
- [ ] Phase 5 — performance engineering
- [ ] Phase 6 — tabula-rasa vs warm-start experiment

## Layout

```
crates/az-core     game abstraction, encodings, Gumbel MCTS, self-play driver
crates/az-perft    perft correctness + movegen throughput benchmark
bench/results      committed benchmark results and the decisions they drove
```

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

Requires a Rust toolchain (`rustup`) and Python 3.14 with PyTorch.

```
cargo build --release
cargo test --release
```

Always benchmark in release. Debug builds are 20–50x slower and any timing from one is
meaningless.

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
