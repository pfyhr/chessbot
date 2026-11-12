# Lc0's training data decodes cleanly into our engine; our replay buffer is the fat one

**Date:** 2026-11-12. Desk research plus a decode check on 300 test80 games.
Nothing here is a strength measurement.

## Why look at it

At 201k self-play evals/s on one 3090, AlphaZero's 44M games at our settings
(32 sims, ~150 plies) cost about 2x10^11 evaluations, which is **~2 weeks** of
one card. That buys AlphaZero's *experience*, not its strength: a 6x96 net
probably plateaus well short of it (a guess from Lc0's small nets, not a
measurement). An AlphaZero-sized 20x256 net costs ~24x per eval
((20/6) x (256/96)^2), so **~300 days on one 3090**, or roughly 30-45 days
on 3-4 RTX 5090s. Lc0 needed far more games than AlphaZero, so 44M is a floor.
The 32-sims-learns-like-800 assumption is from the Gumbel paper at small
budgets and is unproven for chess at scale.

A warm start from Lc0's public data could skip most of that. Lc0's later nets
(test80, the BT transformers) were themselves trained largely on existing,
rescored data.

## What exists

Nets (blocks x channels): the 11258 distilled family from 2x16 to 10x128
(`11258-96x8` is closest to our 6x96), ~10x128 run nets (703810, the last
T70 net), classic 20x256 and larger, and BT4 transformers (15 layers, 1024
wide, ~4 GB). Their weights cannot be loaded into ours: 112 input planes
against our 119, 1858 policy entries against our 4672.

Training data, summed from the storage.lczero.org directory listings:

| run | compressed |
|---|---:|
| test80 | 9.4 TB |
| test75 | 6.1 TB |
| test60 | 5.7 TB |
| test79 | 2.7 TB |
| test91 | 1.7 TB |
| test40 / test30 | 1.4 / 1.1 TB |
| **all** | **~28 TB** |

test78 and test90 list files but every archive is empty. The newest non-empty
test80 data is August 2025. One sampled archive gave **~200 bytes per position**
compressed (gzip ~42x, ~108 positions per game), so test80 holds roughly
47 billion positions in ~430 million games. 20 GB, about 100 million positions,
is plenty for a warm start.

## The format, verified against our core

277 standard games, 31,340 positions, **zero mismatches**: every stored board
equals our replay of the game, every position's set of policy entries equals
our legal-move set exactly, and every played move resolves.

Records are V6, 8,356 bytes, input format 1 (classic 112 planes, no
canonicalisation). What had to be worked out:

- Plane bits are mirrored within each byte: `square = rank*8 + 7 - bit%8`,
  side-to-move view (ranks flipped for Black).
- The 1858 table is every queen-line and knight move ordered by from- then
  to-square (1,792), then 66 q/r/b promotions. A plain 7th-to-8th-rank pawn
  move is the knight promotion.
- Castling is stored king-takes-rook (`e1h1`); ours is `e1g1`.
- Illegal moves carry policy -1.
- `result_q` is from the side to move and alternates sign each ply.
- ~7% of games are Chess960 (our FEN parser rejects them; skip). Every
  standard game starts from the initial position, with no book.
- ~43% of games end by adjudication or resignation, not on the board, so the
  stored result is the target; it cannot be re-derived by replay.

## Storage: theirs against ours

| format | bytes / position |
|---|---:|
| our trainer's replay buffer (fp16 obs + fp16 4672 policy + masks) | ~29,000 |
| Lc0 V6, raw | 8,356 |
| Lc0 V6, gzipped | ~200 |
| start FEN + moves + sparse policy, replayed in Rust | ~150 |

Lc0's raw records are self-contained (random access, no replay) but carry
the 8-ply history and a dense float policy that is mostly -1. Our buffer is
3.5x larger than that and is what limits the replay window today. The
games-plus-moves form is the smallest and costs only a replay through the
Rust encoder, which is microseconds per position.

The 1858 against 4672 policy layout matters little at our size: the extra
slots are a 96->73 1x1 conv and a larger softmax, and the packed bridge
already gathers legal logits on the device. 1858 earns its place with an
attention policy head (one score per from-to pair).

## Plan

Warm-start A/B on Jupiter, same student net and GPU-hours per arm, scored by
fixed-node round robin with the opening book and then by a fixed stretch of
self-play from each:

- **A** supervised on Lc0 positions with their stored search policy and result;
- **B** the same positions labelled by a raw Lc0 net (teacher);
- **C** our own self-play positions labelled by the same teacher.

A against B separates search targets from raw-net targets; B against C
separates the position distribution. A needs no teacher and runs first.
The same games-plus-moves storage replaces the trainer's replay buffer.
