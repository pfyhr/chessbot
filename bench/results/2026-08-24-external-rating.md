# An external rating, and three proxies that lied

**Date:** 2026-08-24 · **Tools:** fastchess (built from source), Stockfish 18

Every rating in this project had been relative — generation N against generation 0, or
against a random mover through our own match code. The UCI engine makes external
measurement possible.

## Round robin, 72 games each, tc=8+0.08

| engine | Elo | score |
|---|---:|---:|
| Stockfish, one node | ∞ | 100.0% |
| chessbot gen145 | **+89 ±117** | 62.5% |
| chessbot gen000 | −255 ±117 | 18.8% |
| random mover | −255 ±112 | 18.8% |

**About 340 Elo above a random mover.**

Two things make the scale credible. An untrained network lands statistically
indistinguishable from random, which is exactly where it belongs. And the random mover
had to be written from scratch, because **no Stockfish configuration is weak enough**:
its Elo floor is 1320 (0–6 against us), and limited to a *single node* it still won 72 of
72, because node-limiting does not touch the part of Stockfish that is strong — it
evaluates every root move with NNUE regardless.

Caveats: ±117 Elo on 72 games, no opening book, and 3–4 timeouts under concurrency. The
340-point gap survives all of that; the exact figure does not.

## The opening mystery, resolved

| | raw policy | after search |
|---|---:|---:|
| 1.e4 | 4.4% | **69.5%** |
| 1.g4 | **61.4%** | 0.1% |
| mass on e4/d4/Nf3/c4 | 4.9% | 69.5% |

The `opening` metric measured the **raw policy**, which genuinely does prefer 1.g4. The
engine plays 1.e4. Four hypotheses — replay window, learning rate, value-target quality,
search width — were spent explaining an instability that did not exist.

This became visible only through a GUI. Adding MultiPV and Lc0-style `info string` lines
so Nibbler could display per-move priors took an afternoon and answered in one screenshot
what a week of metrics had not.

## The recurring mistake

Three times a proxy has sent this project after a problem that was not there:

1. **opening mass** — measured the raw policy, not the engine
2. **vs-prev** — returned exactly 0.500 for 22 generations because the ply cap made every
   game a draw
3. **rising losses and falling material after a resume** — read as a network degrading

The third is the sharpest. Continuing training from a checkpoint showed policy loss rising
2.10 → 2.31, value loss 0.205 → 0.276, and material against random falling +22.3 → +18.5.
All three pointed the same way, and the conclusion drawn was that the resume was corrupting
the network.

A head-to-head against its own starting point put it **+83 Elo ahead** (11W 4L 15D, ±76).
A network that plays more solidly wins less material off a blunderer, and sharper search
targets are harder to fit. The proxies moved the wrong way while the thing they stood in
for moved the right way.

**The rule: when a proxy disagrees with head-to-head play, the proxy is wrong.**

## Genuinely wrong, though

Checkpoints contain only `net.state_dict()` — no optimizer state. So `--init` restarts
AdamW with zeroed moments against an already-converged network, and the replay buffer
starts empty and takes eight generations to fill. That is a real defect and worth fixing;
it simply was not causing the harm attributed to it.

## Protocol bugs found by using a real GUI

Both engines looked correct in isolation.

- **Stockfish was the one flagging** at `st=0.3`, not us. The first "wins" recorded against
  it were forfeits, and were nearly reported as real.
- Our timed searches bet a whole search on an estimated speed. They now deepen in rounds
  and check the clock between them.
- `pending_input()` discarded commands arriving mid-search that were not stop/quit/isready,
  swallowing the `position` and `go` a GUI sends straight after `stop` — which is what
  Nibbler's "Desync" was.
- `nodes` in a multipv line carried per-move visits. UCI means *total* nodes there, and a
  GUI counts it against its own limit, so self-play never advanced.
