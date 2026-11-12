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


def _drive(sp, forward, timing: PoolTiming, until: int | None = None) -> None:
    """Drive one pool until it runs out of games or, with `until`, until its
    cumulative `games_completed` reaches that many."""
    packed = getattr(forward, "packed", False)
    next_batch = sp.next_batch_packed if packed else sp.next_batch
    submit = sp.submit_packed if packed else sp.submit
    while until is None or sp.games_completed < until:
        a = time.perf_counter()
        batch = next_batch()
        timing.driver += time.perf_counter() - a
        if batch is None:
            return
        # `forward` ends in a device-to-host copy, so this is real GPU time.
        a = time.perf_counter()
        logits, values = forward(batch)
        timing.gpu += time.perf_counter() - a
        timing.evals += len(values)
        timing.batches += 1
        a = time.perf_counter()
        submit(logits, values)
        timing.driver += time.perf_counter() - a


def unpack_obs(bits: torch.Tensor, scale: torch.Tensor, shape) -> torch.Tensor:
    """`(N, C)` int64 masks and `(N, C)` scales -> `(N, C, H, W)` float planes.

    Bit i of a mask is square i of its plane, as `SelfPlay::next_batch_packed`
    writes it. An arithmetic right shift is fine for bit 63: `& 1` keeps only
    the bit asked for.
    """
    c, h, w = shape
    shifts = torch.arange(h * w, device=bits.device, dtype=torch.int64)
    planes = ((bits.unsqueeze(-1) >> shifts) & 1).to(scale.dtype) * scale.unsqueeze(-1)
    return planes.view(len(bits), c, h, w)


def make_forward(infer, device: str, bucket: int = 1, cap: int | None = None, amp=None,
                 packed: bool = False):
    """A forward that takes the driver's batch and returns what its submit wants.

    `bucket` pads the batch up to a multiple of itself (capped at `cap`) so a
    compiled network sees a handful of shapes instead of one per batch size.
    `amp` is a dtype for autocast, or None.

    `packed` selects the compact bridge (`next_batch_packed`/`submit_packed`):
    bit-packed planes go up, and only the legal moves' logits come back --
    about 1.4 KB up and 0.15 KB down per position instead of 30 KB and 19 KB.
    """
    ac_type = device.split(":")[0]

    def run_net(x: torch.Tensor, n: int):
        if bucket > 1:
            b = bucket_size(n, bucket, cap or n)
            if b > n:
                x = torch.cat([x, x.new_zeros(b - n, *x.shape[1:])], 0)
        ac = (torch.autocast(device_type=ac_type, dtype=amp)
              if amp is not None else contextlib.nullcontext())
        with torch.no_grad(), ac:
            logits, wdl = infer(x)
        return logits[:n], wdl_to_scalar(wdl[:n].float())

    def forward_dense(obs: np.ndarray):
        logits, values = run_net(torch.from_numpy(obs).to(device), len(obs))
        return (np.ascontiguousarray(logits.float().cpu().numpy()),
                np.ascontiguousarray(values.cpu().numpy().astype(np.float32)))

    def forward_packed(batch):
        bits, scale, legal = batch
        n = len(bits)
        x = unpack_obs(torch.from_numpy(bits.view(np.int64)).to(device),
                       torch.from_numpy(scale).to(device), cc.OBS_SHAPE)
        logits, values = run_net(x, n)
        idx = torch.from_numpy(legal).to(device).long()
        compact = logits.gather(1, idx).float()
        return (np.ascontiguousarray(compact.cpu().numpy()),
                np.ascontiguousarray(values.cpu().numpy().astype(np.float32)))

    if packed:
        forward_packed.packed = True
        return forward_packed
    return forward_dense


class GraphForward:
    """The packed forward as CUDA graphs: one per bucket size, per pool.

    At batch 1024 the network is ~40 kernels and the glue around it a dozen
    more, each a separate launch from Python. A graph replays all of them as
    one. Shapes must be fixed, so batches are padded up to a bucket, exactly
    as for `torch.compile`.

    Everything is captured up front, on the calling thread, before any pool
    starts: capturing (or compiling) from two pool threads at once is what
    hung the first `--compile` run. Each pool owns its graphs, its static
    input buffers and its memory pool, so pools on different threads never
    write over each other's inputs or outputs. The gather of legal logits
    stays outside the graph because its width changes every batch.
    """

    packed = True

    def __init__(self, infer, device: str, *, bucket: int, cap: int, amp, pools: int):
        c, h, w = cc.OBS_SHAPE
        self.device, self.bucket, self.cap = device, bucket, cap
        sizes = sorted({bucket_size(n, bucket, cap) for n in range(1, cap + 1)}, reverse=True)
        ac_type = device.split(":")[0]

        def body(bits, scale):
            x = unpack_obs(bits, scale, cc.OBS_SHAPE)
            ac = (torch.autocast(device_type=ac_type, dtype=amp)
                  if amp is not None else contextlib.nullcontext())
            with torch.no_grad(), ac:
                logits, wdl = infer(x)
            return logits, wdl_to_scalar(wdl.float())

        self.sets = []
        for _ in range(pools):
            mempool = torch.cuda.graph_pool_handle()
            graphs = {}
            for b in sizes:  # largest first, so smaller graphs reuse its memory
                bits = torch.zeros(b, c, dtype=torch.int64, device=device)
                scale = torch.zeros(b, c, dtype=torch.float32, device=device)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):  # cuDNN autotuning and any compile happen here
                        body(bits, scale)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=mempool):
                    logits, values = body(bits, scale)
                graphs[b] = (graph, bits, scale, logits, values)
            self.sets.append(graphs)
        torch.cuda.synchronize()

    def for_pool(self, k: int):
        graphs = self.sets[k]

        def forward(batch):
            bits, scale, legal = batch
            n = len(bits)
            graph, s_bits, s_scale, logits, values = graphs[bucket_size(n, self.bucket, self.cap)]
            s_bits[:n].copy_(torch.from_numpy(bits.view(np.int64)))
            s_scale[:n].copy_(torch.from_numpy(scale))
            graph.replay()
            idx = torch.from_numpy(legal).to(self.device).long()
            compact = logits[:n].gather(1, idx).float()
            return (np.ascontiguousarray(compact.cpu().numpy()),
                    np.ascontiguousarray(values[:n].cpu().numpy().astype(np.float32)))

        forward.packed = True
        return forward


def _drive_all(pools: list, forward, until: list[int | None]) -> list[PoolTiming]:
    timings = [PoolTiming() for _ in pools]
    # A forward with per-pool state (CUDA graphs) hands each pool its own.
    per_pool = getattr(forward, "for_pool", None)
    forwards = [per_pool(k) if per_pool else forward for k in range(len(pools))]
    if len(pools) == 1:
        _drive(pools[0], forwards[0], timings[0], until[0])
        return timings
    errors: list[BaseException] = []

    def target(k: int) -> None:
        try:
            _drive(pools[k], forwards[k], timings[k], until[k])
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
    return timings


def run(pools: list, forward) -> list[PoolResult]:
    """Drive every pool to completion, one thread per pool when there are several."""
    timings = _drive_all(pools, forward, [None] * len(pools))
    return [PoolResult(sp.take_training_data(), dict(sp.stats()), tm)
            for sp, tm in zip(pools, timings)]


# Games per slot in streaming mode: in effect, never run out.
_ENDLESS = 1 << 40
_COUNTS = ("games", "white_wins", "black_wins", "draws", "truncated", "evaluations")


class Stream:
    """Self-play that never drains between generations.

    A drained generation starts every game together and ends when the last one
    does, so its tail runs at a shrinking batch: on a 3090 at 1024 slots the
    mean batch was 423. Here the slots run forever. `generation(n)` drives
    until at least `n` more games have finished and returns those games;
    games still in flight carry over, and the next call continues them --
    under whatever weights the network has by then.

    That last point is the one semantic change from the drained loop: a game
    straddling a training step is played partly by the previous network. Lc0
    and KataGo self-play the same way.
    """

    def __init__(self, *, pools: int, concurrency: int, seed: int, **kw):
        if pools < 1 or pools > concurrency:
            raise ValueError("need 1 <= pools <= concurrency")
        self.pools = [
            cc.ChessSelfPlay(concurrency=c, total_games=c * _ENDLESS,
                             seed=seed + k * SEED_STRIDE, **kw)
            for k, c in enumerate(split(concurrency, pools))
        ]
        self._seen = [dict(sp.stats()) for sp in self.pools]

    def generation(self, games: int, forward) -> tuple[tuple, dict, list[PoolTiming]]:
        """At least `games` newly finished games (split across pools), merged."""
        until = [sp.games_completed + g
                 for sp, g in zip(self.pools, split(games, len(self.pools)))]
        timings = _drive_all(self.pools, forward, until)
        results = []
        for k, (sp, tm) in enumerate(zip(self.pools, timings)):
            now = dict(sp.stats())
            before = self._seen[k]
            delta = {key: now[key] - before[key] for key in _COUNTS}
            plies = now["mean_plies"] * now["games"] - before["mean_plies"] * before["games"]
            delta["mean_plies"] = plies / delta["games"] if delta["games"] else 0.0
            self._seen[k] = now
            results.append(PoolResult(sp.take_training_data(), delta, tm))
        data, stats = merge(results)
        return data, stats, timings


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
