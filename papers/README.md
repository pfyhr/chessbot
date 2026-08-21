# Papers this implementation rests on

- **`gumbel-alphazero-danihelka-2022.pdf`** — Danihelka, Guez, Schrittwieser, Silver,
  *Policy improvement by planning with Gumbel*, ICLR 2022. The search: Gumbel-top-k
  sampling at the root, sequential halving, and the completed-Q improved policy used as
  the training target. Section 3.4 covers the sigma transform; note that the min-max
  rescaling and `value_scale = 0.1` that make it work in practice come from DeepMind's
  `mctx` implementation rather than the paper's main text.

- **`mcts-solver-winands-2008.pdf`** — Winands, Björnsson, Saito, *Monte-Carlo Tree
  Search Solver*. Proven-win / proven-loss propagation: a result the rules guarantee is
  a fact, not a statistic, and must not be averaged with value estimates. Implemented in
  `az-core/src/mcts.rs` as `Edge::proven` / `Search::solve`.

- **`alphazero-silver-2017.pdf`** — Silver et al., *Mastering Chess and Shogi by
  Self-Play with a General Reinforcement Learning Algorithm*. The original: network
  architecture, the 4672-move policy encoding, and the training loop this project
  reimplements.
