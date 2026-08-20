"""Evaluate every checkpoint of a chess run.

The per-generation table in the trainer is what was cheap to compute during
training. This is the fuller picture, run afterwards over saved checkpoints, and
it leads with the metric that actually moves early:

**material against a random opponent.** Win/loss is a poor early signal in chess
-- a weak network cannot force mate, so nearly everything is a draw by the ply
cap and the score sits at 0.5 while the network is plainly improving. Material is
continuous and in-distribution, and it moves as soon as the net stops hanging
pieces.

It is calibrated: random against random reads -0.06. But an untrained network
scores anywhere from -2.8 to +5.3 depending on its initialisation, so early
values are noisy and only large moves mean anything.

    .venv/bin/chess-report runs/chess-v1
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import chessbot_core as cc
import numpy as np
import torch

from . import chess_eval as ev
from .net import Net


def load(path: Path, blocks: int, channels: int, device: str) -> Net:
    planes, h, w = cc.OBS_SHAPE
    net = Net(planes, (h, w), cc.POLICY_LEN, blocks, channels).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--plies", type=int, default=60)
    ap.add_argument("--tactics", type=int, default=200)
    ap.add_argument("--every", type=int, default=1, help="evaluate every Nth checkpoint")
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    paths = sorted(args.run.glob("gen*.pt"))
    if len(paths) < 2:
        raise SystemExit(f"need at least two checkpoints in {args.run}")
    gens = [int(re.search(r"gen(\d+)", p.name).group(1)) for p in paths]

    print(f"{len(paths)} checkpoints in {args.run}\n")
    print("calibrating the material metric...")
    rng = np.random.default_rng(0)
    bal = []
    for g in range(args.games):
        p = cc.Position()
        for _ in range(args.plies):
            moves, out = p.expand()
            if out is not None or not moves:
                break
            p = p.after(moves[rng.integers(len(moves))])
        m = ev.material(p.fen())
        bal.append(m if g % 2 == 0 else -m)
    print(f"  random vs random: {np.mean(bal):+.2f} material (the zero point)\n")

    tactics = ev.build_tactics(args.tactics, args.tactics, seed=11)
    base = ev.baselines(tactics)
    print(
        f"tactical suites: {len(tactics)} positions   "
        f"random baseline mate {base['mate']:.3f}  defend {base['defend']:.3f}\n"
    )

    header = f"{'gen':>4} {'material':>9} {'wins':>6} {'mate':>6} {'defend':>7} {'open':>6}"
    print(header)
    print("-" * len(header))

    rows = []
    for gen, path in zip(gens, paths):
        if gen % args.every and gen != gens[-1]:
            continue
        net = load(path, args.blocks, args.channels, args.device)
        mat = ev.material_vs_random(net, args.games, args.device, seed=1, plies=args.plies)
        tac = ev.tactic_accuracy(net, tactics, args.device)
        opening = ev.opening_mass(net, args.device)
        rows.append({
            "gen": gen, "material": mat["material"], "wins": mat["wins"],
            "mate": tac["mate"], "defend": tac["defend"], "opening": opening,
        })
        print(
            f"{gen:>4} {mat['material']:>+9.2f} {mat['wins']:>6} "
            f"{tac['mate']:>6.3f} {tac['defend']:>7.3f} {opening:>6.3f}"
        )

    first = load(paths[0], args.blocks, args.channels, args.device)
    last = load(paths[-1], args.blocks, args.channels, args.device)
    print("\nlast vs first, randomised openings")
    for sims, label in ((0, "raw policy"), (args.sims, f"search @{args.sims}")):
        r = ev.play_match(last, first, args.games, args.device, seed=99, sims=sims)
        print(
            f"  {label:>14}: {r['wins']}W {r['losses']}L {r['draws']}D  "
            f"score {r['score']:.3f}  ({ev.elo_from_score(r['score']):+.0f} Elo)"
        )

    (args.run / "report.json").write_text(json.dumps(rows, indent=2))
    hist = json.loads((args.run / "history.json").read_text())
    print(
        f"\n{len(hist)} generations, {sum(h['games'] for h in hist):,} games, "
        f"{sum(h['evaluations'] for h in hist):,} evaluations, "
        f"{sum(h['seconds'] for h in hist)/60:.1f} minutes"
    )


if __name__ == "__main__":
    main()
