# Movegen backend decision: shakmaty vs cozy-chess

**Date:** 2026-08-15
**Machine:** Apple M2 Max, 96 GB unified memory, macOS (Darwin 25.5.0)
**Toolchain:** rustc 1.97.1, release profile (`opt-level=3`, `lto="fat"`, `codegen-units=1`)
**Versions:** shakmaty 0.30.1, cozy-chess 0.3.4
**Method:** single-threaded, best-of-5 runs, all six standard CPW perft positions at depth 5.
Correctness verified first: all 48 reference node counts match for both backends.

Reproduce with:

```
cargo build --release
./target/release/az-perft verify --max-depth 4
./target/release/az-perft bench --depth 5 --repeat 5
```

## Result

Mnps = million nodes per second, higher is better. "Ratio" is cozy-chess relative to
shakmaty; below 1.00 means shakmaty is faster.

### make-move mode (no bulk counting) — the mode MCTS actually runs in

| position | shakmaty | cozy-chess | ratio |
|---|---:|---:|---:|
| startpos | **83.0** | 67.0 | 0.81x |
| kiwipete | **100.4** | 78.5 | 0.78x |
| position-3 | **61.7** | 59.4 | 0.96x |
| position-4 | **67.1** | 60.8 | 0.91x |
| position-5 | **85.9** | 83.0 | 0.97x |
| position-6 | **83.5** | 64.7 | 0.78x |

shakmaty wins all six.

### bulk-counting mode — near-pure move enumeration

| position | shakmaty | cozy-chess | ratio |
|---|---:|---:|---:|
| startpos | 200.1 | **245.3** | 1.23x |
| kiwipete | 220.1 | **392.4** | 1.78x |
| position-3 | 118.4 | **256.0** | 2.16x |
| position-4 | 146.1 | **377.2** | 2.58x |
| position-5 | 258.0 | **467.3** | 1.81x |
| position-6 | 216.8 | **528.7** | 2.44x |

cozy-chess wins all six, by a lot.

## Decision: shakmaty

The two modes disagree, and picking the wrong one would have picked the wrong library.

Bulk counting returns the move count at depth 1 without ever making those moves, so it
measures move *enumeration* almost in isolation. cozy-chess is dramatically better at
that — its callback API yields moves grouped per piece as a `PieceMoves` bitboard set,
so a whole piece's moves are produced without materialising them individually. Reading
only that table says "cozy-chess is up to 2.6x faster."

But MCTS does not enumerate in bulk. It expands a node once and then makes exactly one
move per visit, so its cost is dominated by the make-move path — clone the position,
apply the move, update castling/en-passant/check state. On that path shakmaty is
consistently faster, on every position tested, by 3–22%.

The no-bulk column is the one that predicts self-play throughput, so shakmaty wins the
comparison that matters.

Secondary factors, all pointing the same way:

- **Zobrist hashing** built in (`shakmaty::zobrist`) — needed for repetition detection and
  any transposition table.
- **Syzygy tablebases** via `shakmaty-syzygy` — endgame ground truth for training targets.
- **PGN parsing** via `pgn-reader`, same author — needed for the supervised warm-start arm.
- Same author as `python-chess` (niklasf), which is our cross-language test oracle, so the
  two are likely to agree on rules edge cases.
- Powers Lichess in production.

## Caveats

- Single-threaded only. Neither library's threading behaviour was measured; self-play will
  run many games in parallel and that could shift things.
- Both backends clone the position per node rather than using make/unmake with an undo
  stack. An arena-allocated make/unmake scheme could change the absolute numbers
  substantially — though it would change them for both libraries.
- Depth 5 only. Depth 6 on startpos was also measured (shakmaty 62.8 / cozy 63.9 Mnps
  no-bulk, single-shot), which is within the noise band that motivated adding `--repeat`.

## Headline number

`perft(6)` from startpos, single-threaded, full make-move: **119,060,324 nodes in 1.90s**.
The plan's Phase 1 target was "seconds, not minutes." Met.
