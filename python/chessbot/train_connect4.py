"""The Phase 2 gate: does the reinforcement-learning loop actually learn?

Self-play with Gumbel MCTS, train on the improved policy and the game result,
repeat. Connect4 is the subject because a correct loop converges here in minutes,
so a wrong one is obvious rather than being indistinguishable from "chess is
slow".

    .venv/bin/python -m chessbot.train_connect4 --generations 20

What to watch, in order of how much it tells you:

* **tactics** should climb toward 1.0. It is measured against exact ground truth,
  so it cannot be gamed by a degenerate policy.
* **vs-random** should reach ~1.0 early and stay there. If it falls, something
  regressed.
* **centre** should rise toward 1.0 -- Connect4 is solved and the first player
  wins only by taking the middle. Nothing in the loop is told this.
* **vs-prev** near 0.5 means the run has converged, not that it is broken.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import deque
from pathlib import Path

import chessbot_core as cc
import numpy as np
import torch

from . import connect4_eval as ev
from .net import Net, losses, wdl_to_scalar


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def selfplay_generation(net, device: str, args, seed: int):
    """Play a generation of games and return (obs, policy, z) plus stats."""
    net.eval()
    sp = cc.Connect4SelfPlay(
        concurrency=args.concurrency,
        total_games=args.games,
        sims=args.sims,
        max_considered=7,
        seed=seed,
    )

    while (obs := sp.next_batch()) is not None:
        with torch.no_grad():
            logits, wdl = net(torch.from_numpy(obs).to(device))
            values = wdl_to_scalar(wdl.float()).cpu().numpy()
        sp.submit(
            np.ascontiguousarray(logits.float().cpu().numpy()),
            np.ascontiguousarray(values.astype(np.float32)),
        )

    data = sp.take_training_data()
    return data, sp.stats()


def train_steps(net, opt, buffer, args, device: str) -> dict[str, float]:
    """Sample from the replay window and take gradient steps."""
    net.train()
    obs = np.concatenate([b[0] for b in buffer])
    pol = np.concatenate([b[1] for b in buffer])
    z = np.concatenate([b[2] for b in buffer])

    n = len(obs)
    rng = np.random.default_rng(0)
    p_tot = v_tot = 0.0

    for _ in range(args.steps):
        idx = rng.integers(0, n, size=min(args.batch, n))
        x = torch.from_numpy(obs[idx]).to(device)
        pt = torch.from_numpy(pol[idx]).to(device)
        zt = torch.from_numpy(z[idx]).to(device)

        policy_logits, wdl = net(x)
        p_loss, v_loss = losses(policy_logits, wdl, pt, zt)
        loss = p_loss + v_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        p_tot += p_loss.detach().item()
        v_tot += v_loss.detach().item()

    return {
        "policy_loss": p_tot / args.steps,
        "value_loss": v_tot / args.steps,
        "samples": n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--games", type=int, default=256, help="self-play games per generation")
    ap.add_argument("--concurrency", type=int, default=256, help="games in flight (= batch size)")
    ap.add_argument("--sims", type=int, default=32, help="MCTS simulations per move")
    ap.add_argument("--steps", type=int, default=250, help="gradient steps per generation")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--window", type=int, default=5, help="generations kept in the replay buffer")
    ap.add_argument("--eval-games", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/connect4")
    args = ap.parse_args()

    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    planes, h, w = cc.CONNECT4_OBS_SHAPE
    net = Net(planes, (h, w), cc.CONNECT4_POLICY_LEN, args.blocks, args.channels).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    params = sum(p.numel() for p in net.parameters())

    print(f"device={device}  net={args.blocks}x{args.channels} ({params/1e3:.0f}k params)")
    print(f"sims={args.sims}  games/gen={args.games}  concurrency={args.concurrency}\n")

    print("building tactical suite from real play...")
    tactics = ev.build_tactics(400, seed=7)
    kinds = {k: sum(1 for t in tactics if t.kind == k) for k in ("win", "block")}
    print(f"  {len(tactics)} positions with a forced answer: {kinds}\n")

    buffer: deque = deque(maxlen=args.window)
    previous = None
    history = []
    t0 = time.time()

    header = (
        f"{'gen':>4} {'games':>6} {'p_loss':>8} {'v_loss':>8} "
        f"{'tactics':>8} {'win':>6} {'block':>6} {'centre':>7} "
        f"{'vs-rand':>8} {'vs-prev':>8} {'sec':>6}"
    )
    print(header)
    print("-" * len(header))

    for gen in range(args.generations):
        gen_start = time.time()

        (obs, pol, z), stats = selfplay_generation(net, device, args, seed=1234 + gen)
        buffer.append((obs, pol, z))
        losses_ = train_steps(net, opt, buffer, args, device)

        net.eval()
        tac = ev.tactic_accuracy(net, tactics, device)
        centre = ev.centre_preference(net, device)
        vs_random = ev.play_match(
            net, net, args.eval_games, device, seed=gen, opponent_random=True
        )["score"]
        vs_prev = (
            ev.play_match(net, previous, args.eval_games, device, seed=1000 + gen)["score"]
            if previous is not None
            else float("nan")
        )

        elapsed = time.time() - gen_start
        row = {
            "gen": gen,
            "games": stats["games"],
            "mean_plies": stats["mean_plies"],
            "evaluations": stats["evaluations"],
            **losses_,
            "tactics": tac["all"],
            "tactics_win": tac["win"],
            "tactics_block": tac["block"],
            "centre": centre,
            "vs_random": vs_random,
            "vs_prev": vs_prev,
            "seconds": elapsed,
        }
        history.append(row)

        print(
            f"{gen:>4} {stats['games']:>6} {losses_['policy_loss']:>8.4f} "
            f"{losses_['value_loss']:>8.4f} {tac['all']:>8.3f} {tac['win']:>6.3f} "
            f"{tac['block']:>6.3f} {centre:>7.3f} {vs_random:>8.3f} "
            f"{vs_prev:>8.3f} {elapsed:>6.1f}"
        )

        previous = copy.deepcopy(net).eval()
        torch.save(net.state_dict(), out / f"gen{gen:03d}.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))

    total = time.time() - t0
    best = max(history, key=lambda r: r["tactics"])
    print(
        f"\ntotal {total:.0f}s"
        f"\nbest tactics {best['tactics']:.3f} at generation {best['gen']}"
        f"\nfinal vs-random {history[-1]['vs_random']:.3f}"
        f"  centre {history[-1]['centre']:.3f}"
    )


if __name__ == "__main__":
    main()
