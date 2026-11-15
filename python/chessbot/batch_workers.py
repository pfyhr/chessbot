"""Batches from several sampler processes, so the GPU is not waiting on one core.

Replay is single-threaded Python around a Rust call (~90k positions/s on one
aida core); a 6x96 on a 3090 trains at ~100k/s. Each worker owns a disjoint
part of the stream (`StreamSampler(part=(k, n))`) and ships batches in their
storage dtypes, about 7 MB per 4096 positions.
"""

from __future__ import annotations

import multiprocessing as mp
import queue

from chessbot import gamestore


def _worker(k: int, n: int, cfg: dict, q) -> None:
    s = gamestore.StreamSampler(cfg["stream_dir"], window=cfg["window"],
                                capacity=cfg["capacity"], seed=cfg["seed"], part=(k, n))
    while True:
        q.put(s.batch_raw(cfg["batch"]))


class WorkerBatches:
    def __init__(self, stream_dir: str, workers: int, batch: int, capacity: int,
                 window: int = 4, seed: int = 0, prefetch: int = 4):
        cfg = dict(stream_dir=stream_dir, batch=batch, capacity=capacity // workers,
                   window=window, seed=seed)
        ctx = mp.get_context("spawn")  # the parent holds CUDA; children must not fork it
        self.q = ctx.Queue(maxsize=prefetch * workers)
        self.procs = [ctx.Process(target=_worker, args=(k, workers, cfg, self.q), daemon=True)
                      for k in range(workers)]
        for p in self.procs:
            p.start()

    def batch(self) -> dict:
        while True:
            try:
                return gamestore.widen(self.q.get(timeout=60))
            except queue.Empty:
                dead = [p.exitcode for p in self.procs if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"sampler worker died (exit codes {dead})")

    def close(self) -> None:
        for p in self.procs:
            p.terminate()
