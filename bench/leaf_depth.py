"""How deep does the search actually go, with the real network driving it?

`drivercost` answers this with a flat evaluator -- all-zero logits and values --
because its job is to time the driver with the network excluded. That makes its
depth histogram meaningless: with a uniform policy the interior rule
`argmax[pi'(a) - N(a)/(1+sum N)]` degenerates to round-robin, so every child of a
node is opened before any is re-entered, and the tree is as shallow as it can
possibly be. Policy sharpness is exactly what buys depth, so depth has to be
measured with the trained network in the loop.

    .venv/bin/python bench/leaf_depth.py --run runs/champ --sims 32,128,512
    .venv/bin/python bench/leaf_depth.py --run runs/champ --considered 4,8,16,32

Requires a bridge built with the counters on:

    .venv/bin/maturin develop --release -F hotprof
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chessbot_core as cc
import numpy as np
import torch

from chessbot.net import net_from_checkpoint, wdl_to_scalar

MAXD = 9


def latest(run: Path) -> Path:
    paths = sorted(run.glob("gen*.pt"))
    if not paths:
        raise SystemExit(f"no gen*.pt in {run}")
    return paths[-1]


def sweep(net, device, sims, considered, args) -> dict:
    cc.hot_profile_clear()
    sp = cc.ChessSelfPlay(
        concurrency=args.concurrency,
        total_games=args.games,
        sims=sims,
        max_considered=considered,
        max_plies=args.max_plies,
        seed=12345,
    )
    while True:
        obs = sp.next_batch()
        if obs is None:
            break
        with torch.no_grad():
            logits, wdl = net(torch.from_numpy(obs).to(device))
            values = wdl_to_scalar(wdl.float()).cpu().numpy()
        sp.submit(
            np.ascontiguousarray(logits.float().cpu().numpy()),
            np.ascontiguousarray(values.astype(np.float32)),
        )
    c = dict(cc.hot_profile())
    hist = [c.get(f"n_leaf_d{i}", 0) for i in range(1, MAXD)] + [c.get("n_leaf_d9p", 0)]
    leaves = sum(hist)
    moves = c.get("n_commit", 0)
    return {
        "sims": sims,
        "considered": considered,
        "hist": hist,
        "leaves": leaves,
        "moves": moves,
        "evals": c.get("n_evals", 0),
        "mean_depth": c.get("n_leaf_depth_sum", 0) / leaves if leaves else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sims", default="32,64,128,256,512,1024")
    ap.add_argument("--considered", default="16")
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if cc.hot_profile() == []:
        raise SystemExit(
            "the bridge has no counters: rebuild with "
            "`maturin develop --release -F hotprof`"
        )

    ck = latest(args.run)
    planes, h, w = cc.OBS_SHAPE
    net, arch = net_from_checkpoint(ck, planes, (h, w), cc.POLICY_LEN, args.device)
    net.eval()
    print(f"{ck}  {arch}  device={args.device}")
    print(f"{'sims':>6} {'cons':>5} {'moves':>8} {'mean':>6}  " +
          " ".join(f"{'d'+str(i):>7}" for i in range(1, MAXD)) + f" {'d9+':>7}")

    rows = []
    for considered in [int(x) for x in args.considered.split(",")]:
        for sims in [int(x) for x in args.sims.split(",")]:
            r = sweep(net, args.device, sims, considered, args)
            rows.append(r)
            share = [h / r["leaves"] * 100 if r["leaves"] else 0 for h in r["hist"]]
            print(f"{sims:6d} {considered:5d} {r['moves']:8,d} {r['mean_depth']:6.2f}  " +
                  " ".join(f"{s:6.1f}%" for s in share), flush=True)

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
