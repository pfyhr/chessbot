"""Final report for a Connect4 run: does the loop actually produce a better player?

The per-generation table shows `vs-prev`, which only compares neighbours and sits
near 0.5 once a run converges. That is the wrong question for "did this work".
This asks the right ones:

* the last generation against the *first*, head to head
* tactical accuracy with search on, not just raw policy
* both against a random player, as a floor

    PYTHONPATH=python .venv/bin/python -m chessbot.connect4_report runs/connect4-v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chessbot_core as cc
import torch

from . import connect4_eval as ev
from .net import Net


def load(path: Path, blocks: int, channels: int, device: str) -> Net:
    planes, h, w = cc.CONNECT4_OBS_SHAPE
    net = Net(planes, (h, w), cc.CONNECT4_POLICY_LEN, blocks, channels).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    checkpoints = sorted(args.run.glob("gen*.pt"))
    if len(checkpoints) < 2:
        raise SystemExit(f"need at least two checkpoints in {args.run}")

    first = load(checkpoints[0], args.blocks, args.channels, args.device)
    last = load(checkpoints[-1], args.blocks, args.channels, args.device)
    print(f"first = {checkpoints[0].name}   last = {checkpoints[-1].name}\n")

    tactics = ev.build_tactics(400, seed=7)
    kinds = {k: sum(1 for t in tactics if t.kind == k) for k in ("win", "block")}
    print(f"tactical suite: {len(tactics)} forced positions {kinds}\n")

    rows = []
    for name, net in (("first", first), ("last", last)):
        raw = ev.tactic_accuracy(net, tactics, args.device)
        searched = tactics_with_search(net, tactics, args.device, args.sims)
        centre = ev.centre_preference(net, args.device)
        vs_random = ev.play_match(
            net, net, args.games, args.device, seed=1, opponent_random=True
        )["score"]
        rows.append((name, raw, searched, centre, vs_random))

    print(f"{'net':>6} {'tactics(raw)':>13} {'tactics(search)':>16} {'centre':>8} {'vs-random':>10}")
    print("-" * 58)
    for name, raw, searched, centre, vs_random in rows:
        print(
            f"{name:>6} {raw['all']:>13.3f} {searched:>16.3f} {centre:>8.3f} {vs_random:>10.3f}"
        )

    print("\nhead to head, randomised openings")
    for sims, label in ((0, "raw policy"), (args.sims, f"search @{args.sims}")):
        r = ev.play_match(last, first, args.games, args.device, seed=99, sims=sims)
        elo = ev.elo_from_score(r["score"])
        print(
            f"  last vs first, {label:>14}: "
            f"{r['wins']}W {r['losses']}L {r['draws']}D  "
            f"score {r['score']:.3f}  ({elo:+.0f} Elo)"
        )

    history = json.loads((args.run / "history.json").read_text())
    print(
        f"\n{len(history)} generations, "
        f"{sum(h['games'] for h in history):,} self-play games, "
        f"{sum(h['evaluations'] for h in history):,} network evaluations, "
        f"{sum(h['seconds'] for h in history) / 60:.1f} minutes"
    )


def tactics_with_search(net, tactics, device: str, sims: int) -> float:
    """Same suite, but letting the engine actually search."""
    positions = [t.position for t in tactics]
    picks = ev.search_moves(net, positions, device, sims, seed=5)
    return sum(p == t.answer for p, t in zip(picks, tactics)) / len(tactics)


if __name__ == "__main__":
    main()
