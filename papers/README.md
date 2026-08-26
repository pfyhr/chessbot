# Papers this implementation rests on

The PDFs are not redistributed here; each is linked at its canonical source.

### Gumbel AlphaZero
**Danihelka, Guez, Schrittwieser, Silver — _Policy improvement by planning with Gumbel_, ICLR 2022**
<https://openreview.net/forum?id=bERaNdoegnO>

The search: Gumbel-top-k sampling at the root, sequential halving, and the completed-Q
improved policy used as the training target. Implemented in `crates/az-core/src/mcts.rs`.

Worth knowing: the two details that make the sigma transform actually work — min-max
rescaling of completed-Q, and `value_scale = 0.1` — are in DeepMind's
[`mctx`](https://github.com/google-deepmind/mctx) implementation, **not** in the paper's
main text. Getting them wrong does not break training, it just makes it quietly bad; see
`bench/results/2026-08-19-connect4-loop.md`.

### Monte-Carlo Tree Search Solver
**Winands, Björnsson, Saito — _Monte-Carlo Tree Search Solver_, CG 2008**
<https://dke.maastrichtuniversity.nl/m.winands/documents/uctloa.pdf>

Proven-win / proven-loss propagation: a result the rules guarantee is a fact, not a
statistic, and must not be averaged with value estimates. `Edge::proven` and
`Search::solve`. Worth +49 Elo here, and it took mate-in-1 accuracy from 0.22 to 1.00 —
see `bench/results/2026-08-21-proven-win.md`.

### AlphaZero
**Silver et al. — _Mastering Chess and Shogi by Self-Play with a General Reinforcement
Learning Algorithm_, 2017**
<https://arxiv.org/abs/1712.01815>

The original: network architecture, the 4672-move policy encoding, and the training loop
this project reimplements at roughly 0.02% of the compute.
