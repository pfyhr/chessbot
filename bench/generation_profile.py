"""Where does a generation's wall clock actually go?

The renting decision turns on one number, and it is not the GPU's speed. It is
the *fraction of a generation that a faster GPU would touch*. Amdahl does the
rest: if inference and training are 31% of the wall clock, a 6x faster card buys
1.4x more generations per hour, not 6x -- and the money is better spent on the
other 69%, which is our own code and free to fix.

A back-of-envelope on recorded history put that fraction near a third. This
measures it instead, on the same code path training actually uses, with timers
around each phase and around the torch forward inside self-play.

    .venv/bin/python bench/generation_profile.py
    .venv/bin/python bench/generation_profile.py --device cuda --concurrency 1024
    .venv/bin/python bench/generation_profile.py --sweep 256,512,1024,2048

On a borrowed or rented box, run the default first so the number is comparable
to the laptop, then sweep concurrency -- a batch sized for an M2 will not
saturate a 3090, and an idle GPU is still billed.
"""

from __future__ import annotations

import argparse
import contextlib
import time

import chessbot_core as cc
import numpy as np
import torch

from chessbot.net import Net, wdl_to_scalar


# What "fp32" actually means on CUDA, which is not what it sounds like.
#
# torch does not default to one precision. `cuda.matmul.allow_tf32` is False,
# but `cudnn.allow_tf32` is TRUE, and this net is almost all convolution. So a
# CUDA run that sets nothing has been doing TF32 convolutions and fp32 matmuls
# -- a mix, labelled fp32 by whoever wrote it down. Any ranking of cards by
# "fp32 peak" rests on that label being true, so the label has to be checked.
#
#   strict   TF32 off everywhere. The only honest fp32.
#   default  whatever this torch ships. What every past measurement actually ran.
#   tf32     TF32 on everywhere.
#   bf16     TF32 on, plus bf16 autocast for forward and backward.
PRECISIONS = ("strict", "default", "tf32", "bf16")


def set_precision(mode: str, device: str):
    """Pin the numerics and return `(autocast_dtype, what_took_effect)`.

    Reports the flags as torch reports them afterwards, not as they were asked
    for: a setting that silently does not apply on this build is exactly the
    failure this is here to catch.
    """
    amp = None
    if device == "cuda":
        if mode == "strict":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        elif mode in ("tf32", "bf16"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if mode == "bf16":
            amp = torch.bfloat16
        effective = (f"matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32} "
                     f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}")
    else:
        if mode == "bf16":
            amp = torch.bfloat16
        effective = "n/a off CUDA"
    return amp, effective


def autocast(device: str, amp):
    if amp is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device, dtype=amp)


def sync(device: str) -> None:
    """Make the timer measure the GPU, not the queue depth."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def profile_selfplay(net, device: str, args) -> dict:
    """One self-play generation, split into GPU time and driver time.

    `next_batch` and `submit` are the Rust side: tree descent, expansion,
    backup, and the bridge crossing. Everything between them is the network.
    """
    net.eval()
    sp = cc.ChessSelfPlay(
        concurrency=args.concurrency,
        total_games=args.games,
        sims=args.sims,
        max_considered=args.considered,
        max_plies=args.max_plies,
        seed=12345,
        threads=args.driver_threads,
    )
    t_gpu = t_driver = 0.0
    evals = 0
    t0 = time.perf_counter()
    while True:
        a = time.perf_counter()
        obs = sp.next_batch()
        t_driver += time.perf_counter() - a
        if obs is None:
            break

        a = time.perf_counter()
        with torch.no_grad(), autocast(device, args.amp):
            logits, wdl = net(torch.from_numpy(obs).to(device))
            values = wdl_to_scalar(wdl.float()).cpu().numpy()
        sync(device)
        t_gpu += time.perf_counter() - a
        evals += len(obs)

        a = time.perf_counter()
        sp.submit(
            np.ascontiguousarray(logits.float().cpu().numpy()),
            np.ascontiguousarray(values.astype(np.float32)),
        )
        t_driver += time.perf_counter() - a

    obs, policy, z, vmask, pmask = sp.take_training_data()
    total = time.perf_counter() - t0
    return {
        "total": total,
        "gpu": t_gpu,
        "driver": t_driver,
        "other": total - t_gpu - t_driver,
        "evals": evals,
        "pos_per_sec": evals / t_gpu if t_gpu else 0.0,
        "samples": len(obs),
    }


def profile_training(net, device: str, args, samples: int) -> float:
    """Training steps on synthetic batches of the right shape.

    Real targets are not needed: this measures the forward and backward, and a
    cross-entropy against random targets costs exactly what one against real
    targets costs.
    """
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
    planes, h, w = cc.OBS_SHAPE
    x = torch.randn(args.batch, planes, h, w, device=device)
    pt = torch.rand(args.batch, cc.POLICY_LEN, device=device)
    pt = pt / pt.sum(-1, keepdim=True)
    zt = torch.randint(0, 3, (args.batch,), device=device)

    for _ in range(3):  # warm up allocator and autotuner
        with autocast(device, args.amp):
            p, v = net(x)
            loss = (-(pt * torch.log_softmax(p.float(), -1)).sum(-1).mean()
                    + torch.nn.functional.cross_entropy(v.float(), zt))
        loss.backward()
        opt.zero_grad(set_to_none=True)
    sync(device)

    t0 = time.perf_counter()
    for _ in range(args.steps):
        with autocast(device, args.amp):
            p, v = net(x)
            # The losses stay in fp32 under bf16 autocast: a log_softmax over
            # 4672 logits in bf16 has about three significant digits, which is
            # a quality change rather than a speed one, and this is measuring
            # speed.
            loss = (-(pt * torch.log_softmax(p.float(), -1)).sum(-1).mean()
                    + torch.nn.functional.cross_entropy(v.float(), zt))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    sync(device)
    return time.perf_counter() - t0


def report(sp: dict, train_s: float, args, device: str) -> None:
    gen = sp["total"] + train_s
    gpu = sp["gpu"] + train_s          # training is GPU-bound end to end
    non_gpu = gen - gpu
    f = gpu / gen

    print(f"\n{'=' * 62}")
    print(f"generation profile  device={device}  concurrency={args.concurrency}  "
          f"games={args.games}  sims={args.sims}  driver-threads={args.driver_threads}")
    print("=" * 62)
    print(f"{'self-play, network forward':34s} {sp['gpu']:7.1f}s")
    print(f"{'self-play, Rust driver + bridge':34s} {sp['driver']:7.1f}s")
    print(f"{'self-play, python/numpy overhead':34s} {sp['other']:7.1f}s")
    print(f"{'training, ' + str(args.steps) + ' steps':34s} {train_s:7.1f}s")
    print("-" * 62)
    print(f"{'generation total (no eval)':34s} {gen:7.1f}s")
    print(f"{'GPU-bound fraction':34s} {f:7.1%}")
    print(f"{'inference throughput':34s} {sp['pos_per_sec']:7,.0f} pos/sec")
    print(f"{'positions produced':34s} {sp['samples']:7,d}")
    # The number that transfers between machines. Wall clock does not: it is
    # dominated by whichever GPU happens to be attached. Driver seconds are the
    # CPU's half of the generation, and they are what decides whether a rented
    # card would idle.
    print(f"{'driver seconds per generation':34s} {non_gpu:7.1f}s  "
          f"({non_gpu / gen:.0%} of wall clock)")

    print(f"\nAmdahl: what a faster GPU would actually buy at this fraction")
    for s in (2, 4, 6, 10):
        print(f"  a {s:2d}x faster GPU  ->  {1 / ((1 - f) + f / s):.2f}x generations per hour")
    if f < 0.5:
        print("\n  Under half the wall clock is GPU. Buying a faster card is the")
        print("  wrong lever until the rest is cheaper -- and the rest is our code.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--games", type=int, default=64,
                    help="fewer than a real generation; the split is what matters")
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--driver-threads", type=int, default=1,
                    help="self-play driver threads, sharded by game")
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--considered", type=int, default=32)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--sweep", default=None,
                    help="comma-separated concurrencies; reports inference throughput only")
    ap.add_argument("--precision", choices=PRECISIONS, default="default",
                    help="numerics for forward and training; see PRECISIONS")
    ap.add_argument("--tf32", action="store_true",
                    help="deprecated alias for --precision tf32")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the net, as training does by default")
    args = ap.parse_args()
    if args.tf32:
        args.precision = "tf32"

    device = args.device
    if device == "auto":
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    args.amp, effective = set_precision(args.precision, device)

    planes, h, w = cc.OBS_SHAPE
    net = Net(planes, (h, w), cc.POLICY_LEN, args.blocks, args.channels).to(device)
    params = sum(p.numel() for p in net.parameters())
    if args.compile:
        net = torch.compile(net)
    print(f"torch {torch.__version__}  device={device}  net={args.blocks}x{args.channels} "
          f"({params / 1e6:.2f}M params)  compile={args.compile}")
    print(f"precision: {args.precision}  ->  {effective}  autocast={args.amp}")
    if device == "cuda":
        pr = torch.cuda.get_device_properties(0)
        print(f"gpu: {pr.name}  sm_{pr.major}{pr.minor}  {pr.total_memory / 2**30:.0f} GiB")

    if args.sweep:
        x = None
        print(f"\n{'batch':>7s} {'pos/sec':>11s}  {'rel':>6s}")
        base = None
        for b in (int(v) for v in args.sweep.split(",")):
            x = torch.randn(b, planes, h, w, device=device)
            net.eval()
            with torch.no_grad(), autocast(device, args.amp):
                for _ in range(5):
                    net(x)
                sync(device)
                t0, n = time.perf_counter(), 0
                while time.perf_counter() - t0 < 5.0:
                    net(x)
                    n += b
                sync(device)
                r = n / (time.perf_counter() - t0)
            base = base or r
            print(f"{b:>7d} {r:>11,.0f}  {r / base:>5.2f}x")
        print("\nIf throughput is still climbing at the largest batch, the GPU is")
        print("not saturated and self-play concurrency should be raised.")
        return

    sp = profile_selfplay(net, device, args)
    train_s = profile_training(net, device, args, sp["samples"])
    report(sp, train_s, args, device)


if __name__ == "__main__":
    main()
