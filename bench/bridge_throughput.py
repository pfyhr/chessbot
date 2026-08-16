"""How much does the Rust/Python boundary cost, and at what batch size does it stop mattering?

The whole design rests on crossing the boundary once per neural-net batch rather
than once per node. That is only worth doing if the crossing is cheap *relative to
the network forward it wraps*. This measures both sides of that comparison, so the
self-play batch size is chosen on evidence.

    .venv/bin/python bench/bridge_throughput.py
"""

from __future__ import annotations

import time

import chessbot_core as cc
import numpy as np
import torch
import torch.nn as nn

BATCH_SIZES = [1, 8, 32, 128, 512, 1024, 2048]
PLANES = cc.OBS_SHAPE[0]


def sample_positions(n: int, seed: int = 12345) -> list[cc.Position]:
    """Positions from random play, so the mix is not all opening positions."""
    rng = np.random.default_rng(seed)
    out: list[cc.Position] = []
    while len(out) < n:
        pos = cc.Position()
        for _ in range(200):
            out.append(pos)
            if len(out) >= n:
                break
            moves, outcome = pos.expand()
            if outcome is not None or not moves:
                break
            pos = pos.after(moves[rng.integers(len(moves))])
    return out[:n]


def time_it(fn, repeats: int) -> float:
    """Best-of, for the same reason perft uses best-of: least scheduler noise."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


# --- a representative network, sized as planned for the first chess runs -----


class SEBlock(nn.Module):
    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // ratio)
        self.fc2 = nn.Linear(channels // ratio, channels * 2)

    def forward(self, x):
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))
        s = torch.relu(self.fc1(s))
        s = self.fc2(s)
        w, bias = s[:, :c], s[:, c:]
        return x * torch.sigmoid(w).view(b, c, 1, 1) + bias.view(b, c, 1, 1)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(channels)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.se(self.b2(self.c2(y)))
        return torch.relu(x + y)


class Net(nn.Module):
    """6 x 96 SE-ResNet with policy, WDL and moves-left heads."""

    def __init__(self, blocks: int = 6, channels: int = 96):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.policy = nn.Conv2d(channels, 73, 1)
        self.value = nn.Sequential(
            nn.Conv2d(channels, 8, 1), nn.Flatten(), nn.Linear(8 * 64, 128), nn.ReLU()
        )
        self.wdl = nn.Linear(128, 3)
        self.mlh = nn.Linear(128, 1)

    def forward(self, x):
        x = self.tower(self.stem(x))
        p = self.policy(x).flatten(1)
        v = self.value(x)
        return p, self.wdl(v), self.mlh(v)


def main() -> None:
    print(f"Rust/Python bridge throughput   planes={PLANES} policy={cc.POLICY_LEN}\n")

    largest = max(BATCH_SIZES)
    print(f"sampling {largest} positions from random play...")
    pool = sample_positions(largest)

    # --- encode_batch --------------------------------------------------------
    print("\nencode_batch: positions -> (N, C, 8, 8) float32")
    print(f"{'batch':>6} {'ms/call':>10} {'us/pos':>10} {'pos/sec':>14}")
    print("-" * 44)
    encode_us = {}
    for n in BATCH_SIZES:
        chunk = pool[:n]
        reps = max(3, min(200, 20000 // n))
        secs = time_it(lambda: cc.encode_batch(chunk), reps)
        encode_us[n] = secs * 1e6 / n
        print(f"{n:>6} {secs * 1e3:>10.3f} {secs * 1e6 / n:>10.2f} {n / secs:>14,.0f}")

    # --- network forward, for scale ------------------------------------------
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = Net().to(device).eval()
    params = sum(p.numel() for p in net.parameters())
    print(f"\n6x96 SE-ResNet forward on {device}  ({params / 1e6:.2f}M params)")
    print(f"{'batch':>6} {'ms/call':>10} {'us/pos':>10} {'pos/sec':>14} {'bridge %':>10}")
    print("-" * 56)

    for n in BATCH_SIZES:
        x = torch.zeros(n, PLANES, 8, 8, device=device)

        def fwd():
            with torch.no_grad():
                net(x)
            if device == "mps":
                torch.mps.synchronize()

        fwd()  # warm up kernels and allocator
        reps = max(3, min(50, 4000 // n))
        secs = time_it(fwd, reps)
        per_pos = secs * 1e6 / n
        share = 100 * encode_us[n] / (encode_us[n] + per_pos)
        print(
            f"{n:>6} {secs * 1e3:>10.3f} {per_pos:>10.2f} {n / secs:>14,.0f} {share:>9.1f}%"
        )

    print(
        "\nbridge % is the encode cost as a share of encode + forward.\n"
        "Small numbers mean the boundary is not what limits self-play."
    )


if __name__ == "__main__":
    main()
