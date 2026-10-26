"""How spread out is the raw policy, and does a narrow root flatten it?

At --considered 8 only eight root actions are ever visited, so everything
outside the top eight is improved only through the value term in
`softmax(logit + sigma_completed)`. The worry is that the policy collapses onto
a narrow set and stops exploring. This measures that directly on a checkpoint:

    .venv/bin/python bench/policy_entropy.py a.pt b.pt --device mps

Reported per checkpoint, over the same positions:
  entropy     mean Shannon entropy of the legal-masked policy, in nats
  eff-moves   exp(entropy): how many moves the policy is effectively spreading
              over, which is readable against the legal-move count
  top1        mean probability on the single best move
  legal       mean number of legal moves, the ceiling for eff-moves
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import chessbot_core as cc
import numpy as np

from chessbot.chess_eval import policy_probs
from chessbot.net import net_from_checkpoint


def sample_positions(n: int, seed: int, max_plies: int = 120) -> list:
    """Positions from random play -- network-independent, so both checkpoints
    are judged on exactly the same board states."""
    rng = random.Random(seed)
    out, pos = [], cc.Position()
    plies = 0
    while len(out) < n:
        if pos.outcome() is not None or plies >= max_plies:
            pos, plies = cc.Position(), 0
            continue
        if plies >= 8:          # skip the book-opening region
            out.append(pos)
        moves = pos.legal_moves()
        pos = pos.after(rng.choice(moves))
        plies += 1
    return out[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--positions", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    pos = sample_positions(a.positions, a.seed)
    legal = cc.legal_mask_batch(pos).sum(axis=1)
    print(f"{len(pos)} positions from random play, seed {a.seed}")
    print(f"mean legal moves {legal.mean():.1f}\n")
    print(f"{'checkpoint':<34} {'entropy':>8} {'eff-moves':>10} {'top1':>7}")

    for ck in a.checkpoints:
        net, _ = net_from_checkpoint(ck, cc.OBS_PLANES, (8, 8), cc.POLICY_LEN, device=a.device)
        net.eval()
        p = policy_probs(net, pos, a.device)
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
        print(f"{str(ck):<34} {ent.mean():>8.3f} {np.exp(ent.mean()):>10.2f} "
              f"{p.max(axis=1).mean():>7.3f}")


if __name__ == "__main__":
    main()
