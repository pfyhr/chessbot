# One search, two jobs: `RootActions` 8 for play, 32 for self-play

Date: 2026-10-07. The root width that is best for *playing* is not the one that
is best for *training*, and the two are measured separately below. This records
the decision and the evidence on both sides of each choice.

## Why there are two answers

The search does two different jobs. In a match it chooses a move and nothing is
learned. In self-play it also produces the improved policy that becomes the
training target. Width trades against depth for the first job and against move
coverage for the second, so the optima do not have to coincide — and do not.

Worth stating plainly, because it looks like the thing this project avoids: the
candidate set is **not** a human shortlist. `start_root_phases` takes the top `m`
of `gumbel + logit`, which is exactly sampling `m` moves without replacement from
the network's own policy. No move ordering, no captures-first, no opening book.
The Gumbel term is unbounded, so no legal move is excluded — a 1%-prior move
still enters the sample sometimes. The engine is narrowing itself with its own
learned judgement, which is the thing it is supposed to rely on.

## Match play: 8 wins, and both neighbours are worse

gen695 against itself, booked, fixed 128 nodes, only `RootActions` differing.
Positive means the narrower side won. From `bench/depth_study.sh`, 6-7 Oct.

| root width | Elo vs ra32 | games | mean leaf depth at 128 sims |
|---|---|---|---|
| ra4 | −29 ±36 | 300 | — |
| **ra8** | **+58 ±34** | 300 | **4.68** |
| ra16 | +26 ±40 | 300 | 4.14 |
| ra32 (default) | 0 by construction | — | 3.76 |

An interior optimum. `ra4` over-trusts the policy: too few candidates for search
to overturn a mis-ranked move. `ra32` spends the whole budget on breadth and
verifies little — at 32 sims it cannot even complete one round, which is why
self-play sits at mean depth 1.57. `ra8` balances them, and the depth column
tracks the Elo column.

Confidence: only `ra8` clears its own error bar. Three comparisons were run, so a
single false positive at 95% carries roughly 14% risk, but the inverted U across
all three points is harder to get by chance than any one cell.

## Self-play: 32 wins, and the mechanism is coverage

Here width buys *move coverage*, which is what a one-ply policy improvement needs.
Measured at the self-play budget of 32 simulations; the root schedule does not
read the evaluator, so these counts are exact.

| root width | moves evaluated per turn | share of legal moves |
|---|---|---|
| ra4 | 3.93 | 16.2% |
| ra8 | 7.53 | 31.1% |
| ra16 | 14.10 | 58.2% |
| **ra32** | **23.27** | **96.0%** |

Mean legal moves is 24.24, measured. **At `ra32` with 32 simulations, essentially
every legal move gets its own evaluation every turn** — the training target is a
complete one-ply policy improvement over the whole move list. At `ra16` two
moves in five are never looked at; at `ra8`, seven in ten.

Two independent results say this matters:

- **`pw-32` vs `pw-16`, +102 Elo** (2026-08-22, 200 games head to head). Two 10h
  arms run concurrently, differing only in self-play root width, and *both played
  the match at 16 root actions* — so the margin is trained strength, not a search
  setting carried into the test.
- **The mate-in-1 suite, before either arm was trained**: at 16 root actions with
  ~40 legal moves, accuracy caps at **0.367 regardless of budget**, because the
  winning move is often never sampled; at 64 it reaches **1.000**. A move that is
  never evaluated cannot be learned about, and no amount of search fixes it.

Note the earlier `+160` figure is `pw-32` against the *baseline* and bundles
proven-win propagation with the width change. **+102 is the isolated number** and
is the one this decision rests on.

## The decision

| | setting | why | evidence |
|---|---|---|---|
| Match / UCI | **ra8** | no learning happens, so width is pure budget allocation and depth is what is left to buy | +58 ±34, 300 games, one budget, one network |
| Self-play | **ra32** | the search makes the training target, and a move never evaluated cannot be learned about | +102 Elo head to head; 96% vs 58% coverage; mate suite 1.000 vs 0.367 |

## What is not settled

- **Adopting `ra8` changes the instrument.** Every historical Elo number here was
  measured with the engine at `ra32`. The switch has to be dated and both settings
  kept runnable, or old and new results stop comparing.
- **The match result is one budget on one network.** A confirmatory match at a
  different node count is cheap and worth doing before the default changes.
- **Nothing here tests `ra8` in self-play directly.** The case against it is
  coverage plus the `pw-16` result at a neighbouring width, not an A/B at 8. If
  that is ever worth settling it is a wall-clock-matched training run, both arms
  sharing one GPU — see [[ab-arms-must-share-the-machine]].
- **Self-play root width and self-play simulations are separate questions.** This
  says nothing about `sims 32` versus `sims 64`; see
  `2026-10-07-what-search-is-worth.md`.
- **The match optimum is tied to gen695 and has to be re-measured as training
  advances.** Search depth is a function of policy sharpness: a flat policy
  reaches mean depth 2.08 at 128 sims / ra16 where gen695 reaches 4.14, because
  the interior rule re-enters a known child after about `1/pi'_best` visits. A
  sharper network therefore searches deeper at the same budget, and the balance
  that put the peak at 8 moves with it.

  Which way it drifts is **unknown**. Narrowing costs less as ranking improves,
  but it also buys less once depth is already cheap, and those pull opposite
  ways. Re-run the three duels when a new champion is adopted, or roughly every
  300 generations: part 3 of `bench/depth_study.sh` took **2.4h** on one 3090 at
  300 games per pairing, with no retraining. The self-play side does not need
  re-checking on this schedule — it is decided by coverage, not depth.
