# A properly booked rating, and a correction

**Date:** 2026-08-26 · 5 engines, 200 games, `tc=8+0.15`, 240-position 4-ply opening book

| rank | engine | Elo | score | gap above random |
|---|---|---:|---:|---:|
| 1 | Stockfish, one node | +759 | 98.8% | 1518 |
| 2 | **night-g248** (24h training) | **+57 ±93** | 58.1% | **816** |
| 3 | pw32-g145 (10h) | −17 ±86 | 47.5% | 742 |
| 4 | grow-g011 (11h) | −39 ±95 | 44.4% | 720 |
| 5 | random mover | −759 | 1.2% | — |

## The correction

The earlier round robin ran **without an opening book**, and reported the same
`pw32-g145` network as 344 Elo above random. Booked, it is 742.

The cause is visible in the random mover's score: **18.8% unbooked, 1.2% booked**. With
no book and near-deterministic engines, every game from the start position is the same
game. A single line where random happened to survive got replayed dozens of times, which
inflated its score and halved the apparent gap.

So the previously published "~340 Elo above random" understated the engine by more than
2×. The direction was right; the magnitude was not.

## A second correction, less comfortable

The 12-generation resume (`grow-g011`) was reported here as **+83 Elo** over its starting
point `pw32-g145`, and that number was used to argue that the loss and material proxies
had been misleading.

That match was also unbooked. Booked, the two are −39 and −17 with ±86–95 error bars:
**statistically indistinguishable**. The resume neither clearly helped nor clearly hurt.

The proxies still overstated the harm — they said "degrading", the truth is "unchanged".
But the counter-claim of a +83 Elo improvement rested on a measurement no better than the
one it corrected.

## What does hold

`night-g248` is ahead of both predecessors, by 75–96 Elo in the table and by
**+173 ±112** in a separate booked 50-game head-to-head. Two independent booked
measurements agreeing in direction is the strongest evidence in this project so far.

Total training behind that network: roughly 24 hours.

## Still the gap that matters

Stockfish limited to a **single node** sits ~700 Elo above the best network here and
scores 98.8%. It evaluates every root move with NNUE, so node-limiting does not weaken
the part that is strong. There remains no Stockfish configuration weak enough to be a
useful sparring partner.
