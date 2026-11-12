"""Export a chess network for `crates/az-metal`, and a parity fixture to check it.

The file is a flat list of named float32 tensors with BatchNorm already folded
into the preceding convolution (eval-mode statistics), so the native side only
has convolutions, matmuls and elementwise ops to build:

    b"AZMW" u32 version=1 u32 count
    count x { u16 name_len, name (utf-8), u32 ndim, u32 dims[ndim], f32 data }

all little-endian. Linear weights are stored as PyTorch has them, (out, in).

Only the incumbent architecture is supported (residual SE trunk, fc policy
head); anything else is rejected rather than exported wrong.

    .venv/bin/python -m chessbot.export_metal --out net.azmw                 # random init
    .venv/bin/python -m chessbot.export_metal --ckpt runs/x/gen.pt --out net.azmw
    .venv/bin/python -m chessbot.export_metal --out dir/net.azmw --fixture dir

`--fixture` also writes real self-play positions in the packed bridge format
and the PyTorch CPU fp32 outputs for them, which `cargo test -p az-metal`
compares against.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

MAGIC = b"AZMW"
VERSION = 1


def fold_bn(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d):
    """(weight, bias) of the convolution with eval-mode BatchNorm folded in."""
    w = conv.weight.detach().double()
    s = bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)
    b = bn.bias.detach().double() - bn.running_mean.double() * s
    if conv.bias is not None:
        b = b + conv.bias.detach().double() * s
    return (w * s.view(-1, 1, 1, 1)).float(), b.float()


def tensors(net) -> dict[str, torch.Tensor]:
    if net.trunk_kind != "res" or net.head_kind != "fc":
        raise ValueError("az-metal supports the res trunk with the fc policy head only")
    out: dict[str, torch.Tensor] = {}

    def lin(name, m):
        out[f"{name}.w"] = m.weight.detach().float()
        out[f"{name}.b"] = m.bias.detach().float()

    out["stem.w"], out["stem.b"] = fold_bn(net.stem[0], net.stem[1])
    for i, blk in enumerate(net.tower):
        out[f"tower.{i}.c1.w"], out[f"tower.{i}.c1.b"] = fold_bn(blk.c1, blk.b1)
        out[f"tower.{i}.c2.w"], out[f"tower.{i}.c2.b"] = fold_bn(blk.c2, blk.b2)
        lin(f"tower.{i}.se.fc1", blk.se.fc1)
        lin(f"tower.{i}.se.fc2", blk.se.fc2)
    out["policy_conv.w"], out["policy_conv.b"] = fold_bn(net.policy_conv[0], net.policy_conv[1])
    lin("policy_fc", net.policy_fc)
    out["value_conv.w"], out["value_conv.b"] = fold_bn(net.value_conv[0], net.value_conv[1])
    lin("value_fc", net.value_fc[0])
    lin("wdl", net.wdl)
    return out


def write(path, ts: dict[str, torch.Tensor]) -> None:
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<II", VERSION, len(ts)))
        for name, t in ts.items():
            a = np.ascontiguousarray(t.cpu().numpy(), dtype="<f4")
            nb = name.encode()
            f.write(struct.pack("<H", len(nb)) + nb + struct.pack("<I", a.ndim))
            f.write(struct.pack(f"<{a.ndim}I", *a.shape) + a.tobytes())


def fixture(net, out: Path, positions: int, seed: int) -> None:
    """Real packed batches from self-play, plus what PyTorch makes of them."""
    import chessbot_core as cc
    from chessbot import pipeline
    from chessbot.net import wdl_to_scalar

    sp = cc.ChessSelfPlay(concurrency=64, total_games=64, seed=seed,
                          sims=8, max_considered=8, max_plies=200)
    bits, scale, legal = [], [], []
    rng = np.random.default_rng(seed)
    while sum(len(b) for b in bits) < positions:
        batch = sp.next_batch_packed()
        if batch is None:
            break
        b, s, lg = batch
        bits.append(b.copy()), scale.append(s.copy()), legal.append(lg.copy())
        n = len(b)
        sp.submit_packed(rng.standard_normal(lg.shape).astype(np.float32),
                         np.zeros(n, np.float32))
    width = max(lg.shape[1] for lg in legal)
    legal = [np.pad(lg, ((0, 0), (0, width - lg.shape[1]))) for lg in legal]
    bits = np.concatenate(bits)[:positions]
    scale = np.concatenate(scale)[:positions]
    legal = np.concatenate(legal)[:positions].astype(np.int32)

    x = pipeline.unpack_obs(torch.from_numpy(bits.view(np.int64)), torch.from_numpy(scale),
                            cc.OBS_SHAPE)
    with torch.no_grad():
        logits, wdl = net(x)
    out.mkdir(parents=True, exist_ok=True)
    bits.astype("<u8").tofile(out / "bits.u64")
    scale.astype("<f4").tofile(out / "scale.f32")
    legal.astype("<i4").tofile(out / "legal.i32")
    logits.numpy().astype("<f4").tofile(out / "logits.f32")
    wdl.numpy().astype("<f4").tofile(out / "wdl.f32")
    wdl_to_scalar(wdl).numpy().astype("<f4").tofile(out / "value.f32")
    (out / "meta.txt").write_text(f"{len(bits)} {cc.OBS_SHAPE[0]} {width} {cc.POLICY_LEN}\n")


def main() -> None:
    import chessbot_core as cc
    from chessbot.net import Net, net_from_checkpoint

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None, help="checkpoint; random init if omitted")
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixture", default=None, help="directory for a parity fixture")
    ap.add_argument("--positions", type=int, default=512)
    args = ap.parse_args()

    planes, h, w = cc.OBS_SHAPE
    if args.ckpt:
        net, _ = net_from_checkpoint(args.ckpt, planes, (h, w), cc.POLICY_LEN)
    else:
        torch.manual_seed(args.seed)
        net = Net(planes, (h, w), cc.POLICY_LEN, args.blocks, args.channels)
        # Random-init BatchNorm is the identity, which would leave the folding
        # untested. Give it statistics a trained net could have.
        g = torch.Generator().manual_seed(args.seed + 1)
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                c = m.num_features
                m.running_mean.copy_(torch.randn(c, generator=g) * 0.1)
                m.running_var.copy_(torch.rand(c, generator=g) * 0.5 + 0.75)
                m.weight.data.copy_(torch.rand(c, generator=g) * 0.5 + 0.75)
                m.bias.data.copy_(torch.randn(c, generator=g) * 0.1)
    net.eval()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write(args.out, tensors(net))
    print(f"wrote {args.out}")
    if args.fixture:
        fixture(net, Path(args.fixture), args.positions, args.seed)
        print(f"wrote fixture to {args.fixture}")


if __name__ == "__main__":
    main()
