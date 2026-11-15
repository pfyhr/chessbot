"""Supervised training from a game store (e.g. imported Lc0 self-play).

Arm A of the warm-start A/B (bench/results/2026-11-12-lc0-data-as-a-warm-start.md)
and the data/model scaling grid. Checkpoints are the format the self-play
trainer and every match tool read.

The policy loss is the self-play trainer's: soft cross-entropy against a full
softmax over all 4672 slots, so the network still learns to keep illegal
moves at zero. Targets are stored sparsely and gathered on the device.

The value target is a mix: (1 - value_mix) * game result + value_mix * the
search's root WDL. 0 is AlphaZero's target; lc0 found mixing in root values
helps, and the stored root values make it free to try.

Data comes either from shard directories sampled with replacement (`--data`)
or from a one-pass stream (`--stream`, see gamestore.StreamSampler) fed by
`--workers` processes.

Schedules:
  cosine  warmup, then cosine to 0 at --steps (the original arm A schedule)
  wsd     warmup, then constant. At each --branch-at position count a copy of
          the net and its optimiser splits off and decays linearly to 0 over
          the next --branch-frac of that count, trained on the same batches as
          the main run. One pass yields a properly annealed model at every
          data size (Chinchilla's point: an intermediate checkpoint of a long
          schedule is not one), for ~branch-frac extra compute and no extra data.
"""

from __future__ import annotations

import argparse
import copy
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


_DEVICE_DTYPES = {"bits": torch.int64, "scale": torch.float32, "idx": torch.int64,
                  "p": torch.float32, "wdl": torch.float32, "search_wdl": torch.float32}


def to_device(b: dict, device: str, value_mix: float):
    """Batch (storage or widened dtypes) -> device tensors.

    Storage dtypes are half the bytes, and widening them is the GPU's job: on
    the CPU it was most of the main process's time. uint16 policy indices go
    up as int16 (all < 4672) and uint64 masks as int64 (same bits).
    """
    t = {}
    for k, dt in _DEVICE_DTYPES.items():
        v = b[k]
        if v.dtype == np.uint64:
            v = v.view(np.int64)
        elif v.dtype == np.uint16:
            v = v.view(np.int16)
        t[k] = torch.from_numpy(v).to(device, non_blocking=True).to(dt)
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


def wsd_lr(seen: int, peak: float, warmup_positions: int) -> float:
    return peak * min(1.0, (seen + 1) / max(1, warmup_positions))


def branch_lr(seen: int, start: int, length: int, peak: float) -> float:
    return peak * max(0.0, 1.0 - (seen - start) / length)


def parse_counts(s: str | None) -> list[int]:
    return sorted(int(float(x)) for x in s.split(",")) if s else []


def branch_name(label: int) -> str:
    return f"branch_{label // 10**6}M.pt" if label % 10**6 == 0 else f"branch_{label}.pt"


class Branch:
    """A copy of the main run that anneals to zero over `length` positions."""

    def __init__(self, label: int, net, opt, start: int, length: int, compile_: bool):
        self.label, self.start, self.length = label, start, length
        self.net, self.opt = net, opt
        self.fwd = torch.compile(net) if compile_ else net

    @classmethod
    def split(cls, label, net, opt, start, length, compile_, lr):
        bnet = copy.deepcopy(net)
        bopt = torch.optim.AdamW(bnet.parameters(), lr=lr, weight_decay=1e-4)
        bopt.load_state_dict(opt.state_dict())
        return cls(label, bnet, bopt, start, length, compile_)

    def state(self):
        return dict(label=self.label, start=self.start, length=self.length,
                    net=self.net.state_dict(), opt=self.opt.state_dict())


def train_step(fwd, opt, amp, x, idx, p, wdl, lr):
    for g in opt.param_groups:
        g["lr"] = lr
    with amp:
        logits, wl = fwd(x)
    pl, vl = sparse_losses(logits, wl, idx, p, wdl)
    opt.zero_grad(set_to_none=True)
    (pl + vl).backward()
    opt.step()
    return pl, vl


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", nargs="+", help="shard dirs or roots of them (sampled with replacement)")
    src.add_argument("--stream", help="stream dir with order.txt (one pass, see StreamSampler)")
    ap.add_argument("--val", nargs="*", default=None,
                    help="held-out shards (default with --data: the last shard is held out)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None, help="start from this checkpoint's weights")
    ap.add_argument("--resume", action="store_true", help="continue from OUT/resume.pt if it exists")
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--policy-head", choices=["fc", "conv"], default="fc")
    ap.add_argument("--trunk", choices=["res", "attn"], default="res")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--schedule", choices=["cosine", "wsd"], default="cosine")
    ap.add_argument("--warmup", type=int, default=500, help="warmup steps")
    ap.add_argument("--steps", type=int, default=100_000, help="cosine length / step cap")
    ap.add_argument("--branch-at", default=None,
                    help="wsd: comma-separated position counts to anneal a branch at, e.g. 1e8,2e8")
    ap.add_argument("--branch-frac", type=float, default=0.1)
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop at this wall clock (checkpointing first)")
    ap.add_argument("--value-mix", type=float, default=0.0)
    ap.add_argument("--buffer", type=int, default=500_000)
    ap.add_argument("--workers", type=int, default=0, help="--stream: sampler processes (0 = in-process)")
    ap.add_argument("--window", type=int, default=4, help="--stream: shards open per sampler")
    ap.add_argument("--rescan-minutes", type=float, default=0,
                    help="--data: re-scan this often and train on shards that appeared since")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--bf16", action="store_true", help="autocast the forward to bfloat16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    device = pick_device(a.device)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    branch_at = parse_counts(a.branch_at)
    if branch_at and a.schedule != "wsd":
        raise SystemExit("--branch-at needs --schedule wsd")

    # --- data -------------------------------------------------------------
    if a.data:
        shards = gamestore.open_shards(a.data)
        if a.val is None:
            if len(shards) < 2:
                raise SystemExit("need two shards to hold one out, or pass --val")
            train, val = shards[:-1], shards[-1:]
        else:
            train, val = shards, gamestore.open_shards(a.val)
        print(f"train: {len(train)} shards, {sum(s.positions for s in train):,} positions", flush=True)
    else:
        if not a.val:
            raise SystemExit("--stream needs --val")
        val = gamestore.open_shards(a.val)
    print(f"val: {sum(s.positions for s in val):,} positions", flush=True)

    # --- model ------------------------------------------------------------
    if a.init:
        net, arch = net_from_checkpoint(a.init, core.OBS_PLANES, (8, 8), core.POLICY_LEN, device)
    else:
        arch = dict(blocks=a.blocks, channels=a.channels, policy_head=a.policy_head, trunk=a.trunk)
        net = Net(core.OBS_PLANES, (8, 8), core.POLICY_LEN, a.blocks, a.channels,
                  policy_head=a.policy_head, trunk=a.trunk).to(device)
    params = sum(p.numel() for p in net.parameters())
    body = sum(p.numel() for n, p in net.named_parameters() if "policy" not in n and "value" not in n)
    print(f"net: {arch}, {params / 1e6:.2f}M params ({body / 1e6:.2f}M outside the heads), "
          f"device {device}", flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    step, seen, branches, done_branches = 0, 0, [], set()
    resume_path = out / "resume.pt"
    if a.resume and resume_path.exists():
        r = torch.load(resume_path, map_location=device, weights_only=False)
        net.load_state_dict(r["net"])
        opt.load_state_dict(r["opt"])
        step, seen, done_branches = r["step"], r["seen"], set(r["done_branches"])
        for bs in r["branches"]:
            br = Branch.split(bs["label"], net, opt, bs["start"], bs["length"], a.compile, a.lr)
            br.net.load_state_dict(bs["net"])
            br.opt.load_state_dict(bs["opt"])
            branches.append(br)
        print(f"resumed at step {step}, {seen:,} positions, {len(branches)} live branches", flush=True)
    fwd = torch.compile(net) if a.compile else net
    amp = torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=a.bf16)

    t0 = time.time()
    wb = None
    if a.data:
        sampler = gamestore.Sampler(train, capacity=a.buffer, seed=a.seed)
        next_batch = lambda: sampler.batch_raw(a.batch)
    elif a.workers:
        from chessbot.batch_workers import WorkerBatches
        wb = WorkerBatches(a.stream, a.workers, a.batch, a.buffer, window=a.window, seed=a.seed + step)
        next_batch = wb.batch_raw
    else:
        sampler = gamestore.StreamSampler(a.stream, window=a.window, capacity=a.buffer,
                                          seed=a.seed + step)
        next_batch = lambda: sampler.batch_raw(a.batch)
    val_sampler = gamestore.Sampler(val, capacity=min(a.buffer, 50_000), seed=a.seed + 1)
    val_batches = [val_sampler.batch(1024) for _ in range(8)]
    print(f"buffers filled in {time.time() - t0:.0f}s", flush=True)

    meta = dict(arch=arch, args=vars(a), source="train_supervised")
    log = open(out / "log.jsonl", "a")

    def write_row(row):
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in row.items()), flush=True)

    def save_resume():
        tmp = out / "resume.pt.tmp"
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), step=step, seen=seen,
                        done_branches=sorted(done_branches),
                        branches=[br.state() for br in branches], arch=arch), tmp)
        os.replace(tmp, resume_path)

    # With branches, the main run is only needed until the last branch splits off.
    main_end = branch_at[-1] if branch_at else None
    t0 = time.time()
    seen0 = seen
    rate_t, rate_seen = time.time(), seen  # speed since the last eval row
    lr = a.lr
    val_paths = {s.path for s in val}
    last_scan = time.time()
    run = {"policy": 0.0, "value": 0.0, "n": 0}
    while True:
        hours = (time.time() - t0) / 3600
        out_of_time = a.max_hours is not None and hours >= a.max_hours
        main_live = seen < main_end if main_end is not None else step < a.steps
        if out_of_time or (not main_live and not branches):
            break

        if a.data and a.rescan_minutes and time.time() - last_scan > 60 * a.rescan_minutes:
            last_scan = time.time()
            known = val_paths | {s.path for s in sampler.shards}
            found = gamestore.open_shards(a.data, skip=known, allow_empty=True)
            if (k := sampler.add_shards(found)):
                print(f"step={step} added {k} shards, now {len(sampler.shards)}", flush=True)

        b = next_batch()
        x, idx, p, wdl = to_device(b, device, a.value_mix)
        if main_live:
            if a.schedule == "wsd":
                lr = wsd_lr(seen, a.lr, a.warmup * a.batch)
            else:
                lr = a.lr * min(1.0, (step + 1) / a.warmup) * 0.5 * (
                    1 + math.cos(math.pi * min(step, a.steps) / a.steps))
            pl, vl = train_step(fwd, opt, amp, x, idx, p, wdl, lr)
            # Accumulate on the device: .item() here would sync every step.
            run["policy"] += pl.detach()
            run["value"] += vl.detach()
            run["n"] += 1
        for br in branches:
            train_step(br.fwd, br.opt, amp, x, idx, p, wdl,
                       branch_lr(seen, br.start, br.length, a.lr))
        step += 1
        seen += a.batch

        # Split new branches off the main run.
        for c in branch_at:
            if c not in done_branches and seen >= c and all(br.label != c for br in branches):
                branches.append(Branch.split(c, net, opt, seen, int(a.branch_frac * c),
                                             a.compile, a.lr))
                print(f"step={step} branch {c:,} split at {seen:,} positions", flush=True)

        # Finish branches that have annealed.
        for br in [br for br in branches if seen >= br.start + br.length]:
            ev = evaluate(br.net, val_batches, device, a.value_mix)
            write_row(dict(kind="branch", branch=br.label, step=step, positions=seen,
                           hours=round((time.time() - t0) / 3600, 4),
                           val_policy=ev["policy"], val_value=ev["value"], val_top1=ev["top1"]))
            save_checkpoint(out / branch_name(br.label), br.net, br.opt,
                            dict(meta, step=step, positions=seen, branch=br.label))
            branches.remove(br)
            done_branches.add(br.label)

        if main_live and step % a.eval_every == 0:
            ev = evaluate(net, val_batches, device, a.value_mix)
            write_row(dict(kind="main", step=step, positions=seen, hours=round(hours, 4), lr=lr,
                           train_policy=float(run["policy"]) / max(1, run["n"]),
                           train_value=float(run["value"]) / max(1, run["n"]),
                           val_policy=ev["policy"], val_value=ev["value"], val_top1=ev["top1"],
                           pos_per_s=(seen - rate_seen) / (time.time() - rate_t),
                           branches=len(branches)))
            run = {"policy": 0.0, "value": 0.0, "n": 0}
            rate_t, rate_seen = time.time(), seen
        if step % a.save_every == 0:
            meta.update(step=step, positions=seen)
            if main_live:
                save_checkpoint(out / f"step{step}.pt", net, opt, meta)
                save_checkpoint(out / "latest.pt", net, opt, meta)
            save_resume()

    meta.update(step=step, positions=seen)
    if main_end is None:
        ev = evaluate(net, val_batches, device, a.value_mix)
        write_row(dict(kind="main", step=step, positions=seen,
                       hours=round((time.time() - t0) / 3600, 4), lr=lr,
                       val_policy=ev["policy"], val_value=ev["value"], val_top1=ev["top1"],
                       pos_per_s=(seen - seen0) / (time.time() - t0)))
        save_checkpoint(out / f"step{step}.pt", net, opt, meta)
        save_checkpoint(out / "latest.pt", net, opt, meta)
    save_resume()
    print(f"stopped: step {step}, {seen:,} positions, {len(branches)} branches unfinished", flush=True)
    if wb is not None:
        wb.close()


if __name__ == "__main__":
    main()
