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
from .net import Net, load_checkpoint, losses, save_checkpoint, wdl_to_scalar


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def bucket_size(n: int, step: int, cap: int) -> int:
    """Round a batch up to the next multiple of `step`.

    Self-play hands the network a different batch size on almost every call --
    mean 108, median 82, hundreds of distinct values -- because a game sits out
    a batch whenever its leaf is terminal or its search has just finished. That
    variation is what stops `torch.compile` fusing anything: with dynamic shapes
    it measured *slower* than eager, while a fixed shape measured 2.22x faster.

    Rounding to a handful of shapes costs ~1.27x in padded positions that get
    computed and thrown away, and buys the fusion. Measured net: 2.08x on the
    positions that actually count.
    """
    return min(cap, ((n + step - 1) // step) * step)


def selfplay_generation(net, device: str, args, seed: int, infer=None):
    """`infer` is the handle used for forward passes -- a compiled wrapper when
    one is available, otherwise `net` itself. Both share parameters, so weights
    updated by training are seen here without any copying."""
    net.eval()
    infer = infer if infer is not None else net
    sp = cc.ChessSelfPlay(
        concurrency=args.concurrency,
        total_games=args.games,
        sims=args.sims,
        max_considered=args.considered,
        max_plies=args.max_plies,
        seed=seed,
        pcr_prob=args.pcr_prob,
        fast_sims=args.fast_sims,
        fast_considered=args.fast_considered,
    )

    step = args.batch_bucket
    while (obs := sp.next_batch()) is not None:
        n = len(obs)
        x = torch.from_numpy(obs).to(device)
        if step > 1:
            b = bucket_size(n, step, args.concurrency)
            if b > n:
                x = torch.cat([x, x.new_zeros(b - n, *x.shape[1:])], 0)
        with torch.no_grad():
            logits, wdl = infer(x)
        # Drop the padding rows before they reach the driver.
        logits, wdl = logits[:n], wdl[:n]
        values = wdl_to_scalar(wdl.float()).cpu().numpy()
        sp.submit(
            np.ascontiguousarray(logits.float().cpu().numpy()),
            np.ascontiguousarray(values.astype(np.float32)),
        )

    obs, policy, z, mask, pmask = sp.take_training_data()
    # float16 in the buffer: full precision here would be ~4 GB per generation.
    return (obs.astype(np.float16), policy.astype(np.float16), z, mask, pmask), sp.stats()


def train_steps(net, opt, buffer, args, device: str) -> dict[str, float]:
    net.train()
    obs = np.concatenate([b[0] for b in buffer])
    pol = np.concatenate([b[1] for b in buffer])
    z = np.concatenate([b[2] for b in buffer])
    vmask = np.concatenate([b[3] for b in buffer])
    pmask = np.concatenate([b[4] for b in buffer])

    n = len(obs)
    rng = np.random.default_rng(0)
    p_tot = v_tot = 0.0

    for _ in range(args.steps):
        idx = rng.integers(0, n, size=min(args.batch, n))
        x = torch.from_numpy(obs[idx].astype(np.float32)).to(device)
        pt = torch.from_numpy(pol[idx].astype(np.float32)).to(device)
        zt = torch.from_numpy(z[idx]).to(device)
        mt = None if args.no_value_mask else torch.from_numpy(vmask[idx]).to(device)
        # Only masked when PCR is on; otherwise every position is a policy target
        # and passing a mask of ones would only cost a multiply.
        pm = torch.from_numpy(pmask[idx]).to(device) if args.pcr_prob > 0 else None

        policy_logits, wdl = net(x)
        p_loss, v_loss = losses(policy_logits, wdl, pt, zt, mt, pm)
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
        "known_fraction": float(vmask.mean()),
        "policy_target_fraction": float(pmask.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop after this much wall clock, whatever the generation count")
    ap.add_argument("--eval-every", type=int, default=1,
                    help="run the full evaluation every Nth generation")
    ap.add_argument("--no-value-mask", action="store_true",
                    help="train the value head on truncated games too, scoring them "
                         "as draws -- the pre-fix behaviour, kept so it can be "
                         "measured against rather than assumed worse")
    ap.add_argument("--games", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--considered", type=int, default=16)
    # 400, not 200: at 200 nearly half of all games ran out of plies and 55% of
    # training positions carried an invented result. At 400 only 2% truncate, for
    # ~16% more compute -- the games were finishing, they just needed room.
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch", type=int, default=512)
    # 5e-4 and a window of 8, not 1e-3 and 4: at the old settings opening
    # preference oscillated between 0.111 and 0.730 across generations, which is
    # a network chasing whatever it saw last rather than one learning slowly.
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--channels", type=int, default=96)
    # Playout cap randomization. The full budget must be large enough that
    # sequential halving actually runs: at max_considered m, the first phase
    # alone costs m simulations, so sims must exceed m for a second phase to
    # exist at all. At --sims 32 --considered 32 it never does.
    ap.add_argument("--pcr-prob", type=float, default=0.0,
                    help="fraction of moves given the full budget and used as "
                         "policy targets; 0 disables PCR entirely")
    ap.add_argument("--fast-sims", type=int, default=8,
                    help="simulations for the cheap majority of moves")
    ap.add_argument("--fast-considered", type=int, default=8,
                    help="root actions for cheap moves; must not exceed "
                         "--fast-sims or some contenders never get a visit")
    ap.add_argument("--compile", dest="compile", action="store_true", default=True,
                    help="fuse the forward with torch.compile (default on)")
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--batch-bucket", type=int, default=64,
                    help="round self-play batches up to a multiple of this so "
                         "torch.compile sees a handful of shapes rather than "
                         "hundreds; 1 disables padding")
    ap.add_argument("--trunk", choices=["res", "attn"], default="res",
                    help="attn puts global mixing in every trunk block instead "
                         "of only in the flat output layer")
    ap.add_argument("--policy-bottleneck", type=int, default=32,
                    help="channels the flat head squeezes through before its "
                         "Linear; 8 cuts that layer from 9.57M to 2.39M params")
    ap.add_argument("--policy-head", choices=["fc", "conv"], default="fc",
                    help="conv is AlphaZero's: 73 planes from a convolution "
                         "rather than one Linear holding 87%% of the parameters")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--eval-games", type=int, default=60)
    ap.add_argument("--tactics", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/chess")
    ap.add_argument("--init", type=Path, default=None,
                    help="start from this checkpoint instead of random weights")
    args = ap.parse_args()

    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    planes, h, w = cc.OBS_SHAPE
    net = Net(planes, (h, w), cc.POLICY_LEN, args.blocks, args.channels,
              policy_head=args.policy_head, trunk=args.trunk,
              policy_bottleneck=args.policy_bottleneck).to(device)
    if args.init:
        # Continuing beats restarting: every run so far has thrown away the
        # hours before it, and the network is the only thing worth keeping.
        net_state, opt_state, _ = load_checkpoint(args.init, device)
        net.load_state_dict(net_state)
        print(f"resuming from {args.init}"
              + ("" if opt_state else "  (no optimizer state -- older checkpoint)"))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.init:
        _, opt_state, _ = load_checkpoint(args.init, device)
        if opt_state is not None:
            opt.load_state_dict(opt_state)
    params = sum(p.numel() for p in net.parameters())

    # One compiled handle, created once and reused for every generation: the
    # fusion is cached per shape, so only the first generation pays for it.
    # It shares parameters with `net`, so training updates are picked up here.
    infer = net
    if args.compile:
        try:
            infer = torch.compile(net)
        except Exception as e:  # a backend that cannot compile is not fatal
            print(f"torch.compile unavailable ({type(e).__name__}), running eager")
            infer = net

    print(f"device={device}  net={args.blocks}x{args.channels} ({params/1e6:.2f}M params)"
          + (f"  pcr={args.pcr_prob:.2f} full={args.sims}/{args.considered} "
             f"fast={args.fast_sims}/{args.fast_considered}" if args.pcr_prob > 0 else "")
          + "\n"
          f"  trunk={args.trunk}  policy_head={args.policy_head}"
          f"  bottleneck={args.policy_bottleneck}")
    print(f"sims={args.sims}  games/gen={args.games}  max_plies={args.max_plies}")
    print(f"window={args.window}  lr={args.lr}  "
          f"value_mask={'off' if args.no_value_mask else 'on'}"
          + (f"  budget={args.max_hours}h" if args.max_hours else "") + "\n")

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
        f"{'trunc':>6} {'mate':>6} {'defend':>7} {'open':>6} {'mat':>6} "
        f"{'vs-rand':>8} {'vs-prev':>8} {'sec':>6}"
    )
    print(header)
    print("-" * len(header))

    last_eval = {}
    for gen in range(args.generations):
        if args.max_hours is not None and (time.time() - t0) / 3600 >= args.max_hours:
            print(f"\nwall-clock budget of {args.max_hours}h reached at generation {gen}")
            break
        gen_start = time.time()

        (obs, pol, z, vmask, pmask), stats = selfplay_generation(
            net, device, args, seed=7000 + gen, infer=infer
        )
        buffer.append((obs, pol, z, vmask, pmask))
        losses_ = train_steps(net, opt, buffer, args, device)

        net.eval()
        # Evaluation is not free -- skipping it on most generations buys back
        # time for the thing being measured.
        full_eval = (gen % args.eval_every == 0) or (gen == args.generations - 1)
        if full_eval:
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
            last_eval = dict(tac=tac, opening=opening, mat=mat,
                             vs_random=vs_random, vs_prev=vs_prev)
        else:
            tac, opening, mat = last_eval["tac"], last_eval["opening"], last_eval["mat"]
            vs_random, vs_prev = last_eval["vs_random"], last_eval["vs_prev"]

        elapsed = time.time() - gen_start
        history.append({
            "gen": gen, "games": stats["games"], "mean_plies": stats["mean_plies"],
            "evaluations": stats["evaluations"], "white_wins": stats["white_wins"],
            "black_wins": stats["black_wins"], "draws": stats["draws"],
            "truncated": stats["truncated"], "elapsed_total": time.time() - t0,
            "evaluated": full_eval, **losses_,
            "mate": tac["mate"], "defend": tac["defend"], "tactics": tac["all"],
            "opening": opening, "material": mat["material"],
            "vs_random": vs_random, "vs_prev": vs_prev,
            "seconds": elapsed,
        })

        print(
            f"{gen:>4} {stats['games']:>6} {stats['mean_plies']:>6.0f} "
            f"{losses_['policy_loss']:>8.4f} {losses_['value_loss']:>7.4f} "
            f"{100*stats['truncated']/max(stats['games'],1):>5.0f}% "
            f"{tac['mate']:>6.3f} {tac['defend']:>7.3f} {opening:>6.3f} "
            f"{mat['material']:>+6.1f} {vs_random:>8.3f} {vs_prev:>8.3f} {elapsed:>6.1f}"
        )

        previous = copy.deepcopy(net).eval()
        save_checkpoint(out / f"gen{gen:03d}.pt", net, opt, {"gen": gen})
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
