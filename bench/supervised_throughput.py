"""How fast can supervised training go, and what limits it: the sampler or the GPU?

Two independent ceilings:
  sampler  positions/s one Sampler (one core) can replay and hand out
  gpu      positions/s of the training step alone, on one batch reused
           (unpack + forward + loss + backward + AdamW), per net size,
           batch size, eager/compiled, fp32/bf16

Usage: python bench/supervised_throughput.py --data DIR [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import time

import torch

import chessbot_core as core
from chessbot import gamestore
from chessbot.net import Net
from chessbot.train_chess import pick_device
from chessbot.train_supervised import sparse_losses, to_device


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def bench_sampler(shards, batch, seconds=20.0):
    s = gamestore.Sampler(shards, capacity=200_000, seed=0)
    n, t0 = 0, time.time()
    while time.time() - t0 < seconds:
        s.batch(batch)
        n += batch
    return n / (time.time() - t0)


def bench_gpu(b, device, blocks, channels, batch, compile_, bf16, steps=60, warm=15):
    net = Net(core.OBS_PLANES, (8, 8), core.POLICY_LEN, blocks, channels).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    fwd = torch.compile(net) if compile_ else net
    amp = torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=bf16)
    x, idx, p, wdl = to_device(b, device, 0.0)

    def step():
        with amp:
            logits, wl = fwd(x)
        pl, vl = sparse_losses(logits, wl, idx, p, wdl)
        opt.zero_grad(set_to_none=True)
        (pl + vl).backward()
        opt.step()

    for _ in range(warm):
        step()
    sync(device)
    t0 = time.time()
    for _ in range(steps):
        step()
    sync(device)
    mem = torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else 0.0
    return batch * steps / (time.time() - t0), mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sizes", default="6x96,10x128,15x192")
    ap.add_argument("--batches", default="1024,2048,4096")
    ap.add_argument("--modes", default="eager,bf16,compile+bf16")
    a = ap.parse_args()
    device = pick_device(a.device)
    shards = gamestore.open_shards(a.data)

    rows = []
    rate = bench_sampler(shards, 1024)
    rows.append(dict(kind="sampler", batch=1024, pos_per_s=round(rate)))
    print(json.dumps(rows[-1]), flush=True)

    big = gamestore.Sampler(shards, capacity=8192, seed=1)
    for size in a.sizes.split(","):
        blocks, channels = map(int, size.split("x"))
        for batch in map(int, a.batches.split(",")):
            b = big.batch(batch)
            for mode in a.modes.split(","):
                torch.manual_seed(0)
                if device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                rate, mem = bench_gpu(b, device, blocks, channels, batch,
                                      "compile" in mode, "bf16" in mode)
                rows.append(dict(kind="gpu", net=size, batch=batch, mode=mode,
                                 pos_per_s=round(rate), peak_gib=round(mem, 2)))
                print(json.dumps(rows[-1]), flush=True)
                torch._dynamo.reset()


if __name__ == "__main__":
    main()
