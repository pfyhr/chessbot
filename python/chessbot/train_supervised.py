"""Supervised training from a game store (e.g. imported Lc0 self-play).

Arm A of the warm-start A/B (bench/results/2026-11-12-lc0-data-as-a-warm-start.md):
learn the stored search policy and the game result, then hand the checkpoint
to `train_chess.py --init` for self-play. Checkpoints are the same format the
self-play trainer and every match tool read.

The policy loss is the self-play trainer's: soft cross-entropy against a full
softmax over all 4672 slots, so the network still learns to keep illegal
moves at zero. Targets are stored sparsely and gathered on the device.

The value target is a mix: (1 - value_mix) * game result + value_mix * the
search's root WDL. 0 is AlphaZero's target; lc0 found mixing in root values
helps, and the stored root values make it free to try.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import chessbot_core as core
from chessbot import gamestore
from chessbot.net import Net, net_from_checkpoint, save_checkpoint
from chessbot.pipeline import unpack_obs
from chessbot.train_chess import pick_device


def sparse_losses(policy_logits, wdl_logits, idx, p, wdl_target):
    logp = F.log_softmax(policy_logits.float(), dim=-1).gather(1, idx)
    policy = -(p * logp).sum(1).mean()
    value = -(wdl_target * F.log_softmax(wdl_logits.float(), dim=-1)).sum(1).mean()
    return policy, value


def to_device(b: dict, device: str, value_mix: float):
    t = {k: torch.from_numpy(v).to(device, non_blocking=True) for k, v in b.items()}
    x = unpack_obs(t["bits"], t["scale"], core.OBS_SHAPE)
    wdl = (1 - value_mix) * t["wdl"] + value_mix * t["search_wdl"]
    p = t["p"] / t["p"].sum(1, keepdim=True).clamp(min=1e-6)
    return x, t["idx"], p, wdl


@torch.no_grad()
def evaluate(net, batches, device, value_mix):
    net.eval()
    tot = {"policy": 0.0, "value": 0.0, "top1": 0.0}
    for b in batches:
        x, idx, p, wdl = to_device(b, device, value_mix)
        logits, wl = net(x)
        pl, vl = sparse_losses(logits, wl, idx, p, wdl)
        tot["policy"] += pl.item()
        tot["value"] += vl.item()
        # Does the net's favourite legal move match the target's favourite?
        legal_logits = logits.float().gather(1, idx).masked_fill(p == 0, -1e9)
        tot["top1"] += (legal_logits.argmax(1) == p.argmax(1)).float().mean().item()
    net.train()
    return {k: v / len(batches) for k, v in tot.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", nargs="+", required=True, help="shard dirs or roots of them")
    ap.add_argument("--val", nargs="*", default=None,
                    help="held-out shards (default: the last training shard is held out)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None, help="start from this checkpoint")
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--policy-head", choices=["fc", "conv"], default="fc")
    ap.add_argument("--trunk", choices=["res", "attn"], default="res")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop at this wall clock; the LR schedule assumes --steps")
    ap.add_argument("--value-mix", type=float, default=0.0)
    ap.add_argument("--buffer", type=int, default=500_000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--rescan-minutes", type=float, default=0,
                    help="re-scan --data this often and train on shards that appeared since")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--bf16", action="store_true", help="autocast the forward to bfloat16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    device = pick_device(a.device)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    shards = gamestore.open_shards(a.data)
    if a.val is None:
        if len(shards) < 2:
            raise SystemExit("need two shards to hold one out, or pass --val")
        train, val = shards[:-1], shards[-1:]
    else:
        train, val = shards, gamestore.open_shards(a.val)
    print(f"train: {len(train)} shards, {sum(s.positions for s in train):,} positions; "
          f"val: {sum(s.positions for s in val):,}", flush=True)

    if a.init:
        net, arch = net_from_checkpoint(a.init, core.OBS_PLANES, (8, 8), core.POLICY_LEN, device)
    else:
        arch = dict(blocks=a.blocks, channels=a.channels, policy_head=a.policy_head, trunk=a.trunk)
        net = Net(core.OBS_PLANES, (8, 8), core.POLICY_LEN, a.blocks, a.channels,
                  policy_head=a.policy_head, trunk=a.trunk).to(device)
    params = sum(p.numel() for p in net.parameters())
    print(f"net: {arch}, {params / 1e6:.2f}M params, device {device}", flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup)
        * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    fwd = torch.compile(net) if a.compile else net
    amp = torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=a.bf16)

    t0 = time.time()
    sampler = gamestore.Sampler(train, capacity=a.buffer, seed=a.seed)
    val_sampler = gamestore.Sampler(val, capacity=min(a.buffer, 50_000), seed=a.seed + 1)
    val_batches = [val_sampler.batch(a.batch) for _ in range(8)]
    print(f"buffers filled in {time.time() - t0:.0f}s", flush=True)

    meta = dict(arch=arch, args=vars(a), source="train_supervised")
    log = open(out / "log.jsonl", "a")
    t0 = time.time()
    seen = 0
    val_paths = {s.path for s in val}
    last_scan = time.time()
    run = {"policy": 0.0, "value": 0.0, "n": 0}
    for step in range(1, a.steps + 1):
        b = sampler.batch(a.batch)
        x, idx, p, wdl = to_device(b, device, a.value_mix)
        with amp:
            logits, wl = fwd(x)
        pl, vl = sparse_losses(logits, wl, idx, p, wdl)
        opt.zero_grad(set_to_none=True)
        (pl + vl).backward()
        opt.step()
        sched.step()
        seen += a.batch
        run["policy"] += pl.item()
        run["value"] += vl.item()
        run["n"] += 1

        if a.rescan_minutes and time.time() - last_scan > 60 * a.rescan_minutes:
            last_scan = time.time()
            known = val_paths | {s.path for s in sampler.shards}
            found = gamestore.open_shards(a.data, skip=known, allow_empty=True)
            if (k := sampler.add_shards(found)):
                print(f"step={step} added {k} shards, now {len(sampler.shards)}", flush=True)

        hours = (time.time() - t0) / 3600
        last = step == a.steps or (a.max_hours is not None and hours >= a.max_hours)
        if step % a.eval_every == 0 or last:
            ev = evaluate(net, val_batches, device, a.value_mix)
            row = dict(step=step, positions=seen, hours=round(hours, 4),
                       lr=sched.get_last_lr()[0],
                       train_policy=run["policy"] / run["n"], train_value=run["value"] / run["n"],
                       val_policy=ev["policy"], val_value=ev["value"], val_top1=ev["top1"],
                       pos_per_s=seen / (time.time() - t0))
            run = {"policy": 0.0, "value": 0.0, "n": 0}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in row.items()), flush=True)
        if step % a.save_every == 0 or last:
            meta.update(step=step, positions=seen)
            save_checkpoint(out / f"step{step}.pt", net, opt, meta)
            save_checkpoint(out / "latest.pt", net, opt, meta)
        if last:
            break


if __name__ == "__main__":
    main()
