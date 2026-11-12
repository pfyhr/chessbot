"""What one self-play forward call costs through PyTorch, split into its parts.

`generation_profile.py --sweep` measures the network with the queue kept full:
calls are enqueued back to back and synchronised once. Self-play cannot do
that. Every call waits for its answer before the trees can move, so each one
pays the full latency: host-to-device, launch, the network, device-to-host.

This times, per call and synchronously, at fixed batch sizes:

    sync      a one-element round trip: the fixed cost of asking at all
    net       the network on a batch already on the device, synchronised
    packed    the whole `--bridge packed` forward from `pipeline.make_forward`
    dense     the whole `--bridge dense` forward

Inputs match `az-metal-bench`: random sparse masks, unit scales, legal width 64.

    .venv/bin/python bench/forward_call_bench.py --device mps
"""

from __future__ import annotations

import argparse
import time

import chessbot_core as cc
import numpy as np
import torch

from chessbot import pipeline
from chessbot.net import Net


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def timeit(fn, secs: float) -> tuple[float, int]:
    for _ in range(5):
        fn()
    t0, n = time.perf_counter(), 0
    while time.perf_counter() - t0 < secs:
        fn()
        n += 1
    return (time.perf_counter() - t0) / n, n


def fold_batchnorm(net) -> None:
    """Replace every conv + eval BatchNorm pair with one biased conv, in place."""
    from chessbot.export_metal import fold_bn

    def fold(conv, bn):
        w, b = fold_bn(conv, bn)
        conv.weight.data.copy_(w)
        conv.bias = torch.nn.Parameter(b)
        return torch.nn.Identity()

    net.stem[1] = fold(net.stem[0], net.stem[1])
    for blk in net.tower:
        blk.b1 = fold(blk.c1, blk.b1)
        blk.b2 = fold(blk.c2, blk.b2)
    net.policy_conv[1] = fold(net.policy_conv[0], net.policy_conv[1])
    net.value_conv[1] = fold(net.value_conv[0], net.value_conv[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batches", default="128,256,512,1024,2048")
    ap.add_argument("--secs", type=float, default=3.0)
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--fold-bn", action="store_true",
                    help="fold eval BatchNorm into the convolutions, as az-metal does")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the net, as training does by default")
    args = ap.parse_args()
    dev = args.device
    amp = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]

    planes, h, w = cc.OBS_SHAPE
    torch.manual_seed(0)
    net = Net(planes, (h, w), cc.POLICY_LEN, args.blocks, args.channels).eval()
    if args.fold_bn:
        fold_batchnorm(net)  # on the CPU: MPS has no float64
    net = net.to(dev)
    if args.compile:
        net = torch.compile(net)
    rng = np.random.default_rng(0)
    width = 64

    one = torch.zeros(1, device=dev)
    rt, _ = timeit(lambda: (one + 1).cpu(), args.secs)
    print(f"torch {torch.__version__} device={dev} precision={args.precision} "
          f"net={args.blocks}x{args.channels} compile={args.compile} fold_bn={args.fold_bn}")
    print(f"sync round trip: {rt * 1e3:.3f} ms")
    print(f"{'batch':>6} {'net ms':>8} {'packed ms':>10} {'dense ms':>9} "
          f"{'net pos/s':>10} {'packed pos/s':>13} {'dense pos/s':>12}")

    fwd_p = pipeline.make_forward(net, dev, amp=amp, packed=True)
    fwd_d = pipeline.make_forward(net, dev, amp=amp, packed=False)
    for n in (int(b) for b in args.batches.split(",")):
        bits = (rng.integers(0, 2**63, (n, planes), dtype=np.int64)
                & rng.integers(0, 2**63, (n, planes), dtype=np.int64)
                & rng.integers(0, 2**63, (n, planes), dtype=np.int64)).view(np.uint64)
        scale = np.ones((n, planes), np.float32)
        legal = rng.integers(0, cc.POLICY_LEN, (n, width)).astype(np.int32)
        obs = pipeline.unpack_obs(torch.from_numpy(bits.view(np.int64)),
                                  torch.from_numpy(scale), cc.OBS_SHAPE).numpy()
        x = torch.from_numpy(obs).to(dev)

        def net_only():
            with torch.no_grad(), (torch.autocast(dev.split(":")[0], dtype=amp)
                                   if amp else torch.autocast(dev.split(":")[0], enabled=False)):
                net(x)
            sync(dev)

        t_net, _ = timeit(net_only, args.secs)
        t_p, _ = timeit(lambda: fwd_p((bits, scale, legal)), args.secs)
        t_d, _ = timeit(lambda: fwd_d(obs), args.secs)
        print(f"{n:>6} {t_net * 1e3:>8.2f} {t_p * 1e3:>10.2f} {t_d * 1e3:>9.2f} "
              f"{n / t_net:>10,.0f} {n / t_p:>13,.0f} {n / t_d:>12,.0f}")


if __name__ == "__main__":
    main()
