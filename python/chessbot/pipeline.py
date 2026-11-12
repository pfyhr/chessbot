"""Self-play split into independent pools, each driven by its own thread.

With one pool the CPU and GPU take turns: the driver walks every tree to a
leaf, the GPU evaluates the batch, the driver backs it up. On a 3090 at 1024
concurrent games the two halves measured about equal (94 s driver, 98 s
forward per generation), so half of every generation one of them sits idle.

With two pools the turns interleave: while pool A waits on its forward, pool B
walks and backs up its trees. Both Rust calls (`next_batch`, `submit`) release
the GIL and torch releases it while waiting on the device, so plain Python
threads are enough.

All pools share the default CUDA stream. Per-pool streams were tried and
produced non-finite logits on a 3090 (the driver's sort then panicked); they
were not needed anyway, since the gain is CPU overlapping GPU, not two GPU
streams overlapping each other.

Pools are independent games with independent seeds; nothing about the search
changes. `pools=1` reproduces the single-driver loop exactly, seed included.
"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass, field

import chessbot_core as cc
import numpy as np
import torch

from chessbot.net import wdl_to_scalar

# Pool k > 0 is seeded `seed + k * SEED_STRIDE`, far from any seed a caller
# would pick for a neighbouring generation.
SEED_STRIDE = 1_000_003


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


def split(total: int, parts: int) -> list[int]:
    """`total` divided as evenly as possible; the first parts get the remainder."""
    q, r = divmod(total, parts)
    return [q + (i < r) for i in range(parts)]


@dataclass
class PoolTiming:
    """Per-pool seconds. Across pools these overlap, so they do not sum to wall clock."""
    gpu: float = 0.0
    driver: float = 0.0
    evals: int = 0
    batches: int = 0


@dataclass
class PoolResult:
    data: tuple
    stats: dict
    timing: PoolTiming = field(default_factory=PoolTiming)


def make_pools(*, pools: int, concurrency: int, games: int, seed: int, **kw) -> list:
    """`kw` is passed through to `ChessSelfPlay` (sims, max_considered, ...)."""
    if pools < 1:
        raise ValueError("pools must be >= 1")
    if pools > min(concurrency, games):
        raise ValueError(f"{pools} pools need at least that many games and slots")
    return [
        cc.ChessSelfPlay(concurrency=c, total_games=g, seed=seed + k * SEED_STRIDE, **kw)
        for k, (c, g) in enumerate(zip(split(concurrency, pools), split(games, pools)))
    ]


def _drive(sp, forward, timing: PoolTiming) -> None:
    while True:
        a = time.perf_counter()
        obs = sp.next_batch()
        timing.driver += time.perf_counter() - a
        if obs is None:
            return
        # `forward` ends in a device-to-host copy, so this is real GPU time.
        a = time.perf_counter()
        logits, values = forward(obs)
        timing.gpu += time.perf_counter() - a
        timing.evals += len(obs)
        timing.batches += 1
        a = time.perf_counter()
        sp.submit(logits, values)
        timing.driver += time.perf_counter() - a


def make_forward(infer, device: str, bucket: int = 1, cap: int | None = None, amp=None):
    """A forward that takes the driver's numpy batch and returns what `submit` wants.

    `bucket` pads the batch up to a multiple of itself (capped at `cap`) so a
    compiled network sees a handful of shapes instead of one per batch size.
    `amp` is a dtype for autocast, or None.
    """
    def forward(obs: np.ndarray):
        n = len(obs)
        x = torch.from_numpy(obs).to(device)
        if bucket > 1:
            b = bucket_size(n, bucket, cap or n)
            if b > n:
                x = torch.cat([x, x.new_zeros(b - n, *x.shape[1:])], 0)
        ac = (torch.autocast(device_type=device.split(":")[0], dtype=amp)
              if amp is not None else contextlib.nullcontext())
        with torch.no_grad(), ac:
            logits, wdl = infer(x)
        logits, wdl = logits[:n], wdl[:n]
        values = wdl_to_scalar(wdl.float())
        return (np.ascontiguousarray(logits.float().cpu().numpy()),
                np.ascontiguousarray(values.cpu().numpy().astype(np.float32)))
    return forward


def run(pools: list, forward) -> list[PoolResult]:
    """Drive every pool to completion, one thread per pool when there are several."""
    timings = [PoolTiming() for _ in pools]
    if len(pools) == 1:
        _drive(pools[0], forward, timings[0])
    else:
        errors: list[BaseException] = []

        def target(k: int) -> None:
            try:
                _drive(pools[k], forward, timings[k])
            except BaseException as e:  # surfaced on the main thread below
                errors.append(e)

        threads = [threading.Thread(target=target, args=(k,), name=f"pool{k}")
                   for k in range(len(pools))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
    return [PoolResult(sp.take_training_data(), dict(sp.stats()), tm)
            for sp, tm in zip(pools, timings)]


def merge(results: list[PoolResult]) -> tuple[tuple, dict]:
    """Concatenate pools' training arrays and combine their stats."""
    if len(results) == 1:
        return results[0].data, results[0].stats
    data = tuple(np.concatenate([r.data[i] for r in results]) for i in range(5))
    stats: dict = {}
    for key in ("games", "white_wins", "black_wins", "draws", "truncated", "evaluations"):
        stats[key] = sum(r.stats[key] for r in results)
    games = stats["games"]
    stats["mean_plies"] = (sum(r.stats["mean_plies"] * r.stats["games"] for r in results)
                           / games if games else 0.0)
    return data, stats
