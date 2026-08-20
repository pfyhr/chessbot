"""Phase 3: the same loop, pointed at chess.

Nothing here is new machinery. `az_core::selfplay` is generic over the `Game`
trait, so the driver, the search and the training target are exactly the ones the
Connect4 run validated. What changes is scale: 4672 policy slots instead of 7,
119 input planes instead of 2, and games that run ~160 plies instead of ~30.

    .venv/bin/c4-train-chess --generations 20      # or -m chessbot.train_chess

What to watch:

* **mate** -- a mate in one is on the board; does the raw policy play it? Exact
  ground truth, cross-checked against python-chess.
* **defend** -- the opponent mates next move unless this one stops it. The
  symmetric test and the harder one; it stays near chance far longer.
* **vs-rand** -- the floor.
* **open** -- policy mass on e4/d4/Nf3/c4. Not ground truth, but a cheap read on
  whether the opening has any structure yet.

Memory note: observations are 119x8x8 floats, so a generation of 256 games is
about a gigabyte. The replay buffer holds them as float16 and casts per batch;
moving the buffer into Rust is a Phase 5 job.
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

from . import chess_eval as ev
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
    net.eval()
    sp = cc.ChessSelfPlay(
        concurrency=args.concurrency,
        total_games=args.games,
        sims=args.sims,
        max_considered=args.considered,
        max_plies=args.max_plies,
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

    obs, policy, z = sp.take_training_data()
    # float16 in the buffer: full precision here would be ~4 GB per generation.
    return (obs.astype(np.float16), policy.astype(np.float16), z), sp.stats()


def train_steps(net, opt, buffer, args, device: str) -> dict[str, float]:
    net.train()
    obs = np.concatenate([b[0] for b in buffer])
    pol = np.concatenate([b[1] for b in buffer])
    z = np.concatenate([b[2] for b in buffer])

    n = len(obs)
    rng = np.random.default_rng(0)
    p_tot = v_tot = 0.0

    for _ in range(args.steps):
        idx = rng.integers(0, n, size=min(args.batch, n))
        x = torch.from_numpy(obs[idx].astype(np.float32)).to(device)
        pt = torch.from_numpy(pol[idx].astype(np.float32)).to(device)
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

    return {"policy_loss": p_tot / args.steps, "value_loss": v_tot / args.steps, "samples": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--games", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--considered", type=int, default=16)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--eval-games", type=int, default=60)
    ap.add_argument("--tactics", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/chess")
    args = ap.parse_args()

    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    planes, h, w = cc.OBS_SHAPE
    net = Net(planes, (h, w), cc.POLICY_LEN, args.blocks, args.channels).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    params = sum(p.numel() for p in net.parameters())

    print(f"device={device}  net={args.blocks}x{args.channels} ({params/1e6:.2f}M params)")
    print(f"sims={args.sims}  games/gen={args.games}  max_plies={args.max_plies}\n")

    print("building tactical suites from random play...")
    tactics = ev.build_tactics(args.tactics, args.tactics, seed=11)
    kinds = {k: sum(1 for t in tactics if t.kind == k) for k in ("mate", "defend")}
    base = ev.baselines(tactics)
    print(f"  {len(tactics)} positions with exact answers: {kinds}")
    print(
        f"  random-move baseline: mate {base['mate']:.3f}  defend {base['defend']:.3f}"
        f"  (opening 0.200)\n"
    )

    buffer: deque = deque(maxlen=args.window)
    previous = None
    history = []
    t0 = time.time()

    header = (
        f"{'gen':>4} {'games':>6} {'plies':>6} {'p_loss':>8} {'v_loss':>7} "
        f"{'mate':>6} {'defend':>7} {'open':>6} {'mat':>6} {'vs-rand':>8} {'vs-prev':>8} {'sec':>6}"
    )
    print(header)
    print("-" * len(header))

    for gen in range(args.generations):
        gen_start = time.time()

        (obs, pol, z), stats = selfplay_generation(net, device, args, seed=7000 + gen)
        buffer.append((obs, pol, z))
        losses_ = train_steps(net, opt, buffer, args, device)

        net.eval()
        tac = ev.tactic_accuracy(net, tactics, device)
        opening = ev.opening_mass(net, device)
        # The sensitive early signal: win/loss stays at 0.5 for a long time
        # because a weak net cannot force mate, but material moves immediately.
        mat = ev.material_vs_random(net, args.eval_games, device, seed=gen, plies=60)
        vs_random = ev.play_match(
            net, net, args.eval_games, device, seed=gen, opponent_random=True
        )["score"]
        vs_prev = (
            ev.play_match(net, previous, args.eval_games, device, seed=5000 + gen)["score"]
            if previous is not None
            else float("nan")
        )

        elapsed = time.time() - gen_start
        history.append({
            "gen": gen, "games": stats["games"], "mean_plies": stats["mean_plies"],
            "evaluations": stats["evaluations"], "white_wins": stats["white_wins"],
            "black_wins": stats["black_wins"], "draws": stats["draws"], **losses_,
            "mate": tac["mate"], "defend": tac["defend"], "tactics": tac["all"],
            "opening": opening, "material": mat["material"],
            "vs_random": vs_random, "vs_prev": vs_prev,
            "seconds": elapsed,
        })

        print(
            f"{gen:>4} {stats['games']:>6} {stats['mean_plies']:>6.0f} "
            f"{losses_['policy_loss']:>8.4f} {losses_['value_loss']:>7.4f} "
            f"{tac['mate']:>6.3f} {tac['defend']:>7.3f} {opening:>6.3f} "
            f"{mat['material']:>+6.1f} {vs_random:>8.3f} {vs_prev:>8.3f} {elapsed:>6.1f}"
        )

        previous = copy.deepcopy(net).eval()
        torch.save(net.state_dict(), out / f"gen{gen:03d}.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))

    total = time.time() - t0
    best = max(history, key=lambda r: r["mate"])
    print(
        f"\ntotal {total/60:.1f} min"
        f"\nbest mate-in-1 {best['mate']:.3f} at generation {best['gen']}"
        f"\nfinal vs-random {history[-1]['vs_random']:.3f}  opening {history[-1]['opening']:.3f}"
    )


if __name__ == "__main__":
    main()
