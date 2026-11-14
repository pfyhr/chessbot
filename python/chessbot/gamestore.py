"""Games stored as moves, replayed into training positions on demand.

A position costs ~150 bytes here instead of the ~29 KB of float16 planes,
policy and masks the self-play replay buffer holds: the store keeps the start
FEN, the policy index of each move played, a sparse policy over the legal
moves, and the value targets. `chessbot_core.replay_packed` rebuilds every
position of a game (history and repetition planes included) as bitboards in
one call, and the device expands them with `pipeline.unpack_obs`.

A shard is a directory of .npy files, memory-mapped on read:

- `fens`        (G,)     start FEN per game
- `game_off`    (G+1,)   position offsets; game g has positions [off[g], off[g+1])
- `played`      (P-G,)   policy index of the move played from each position
                         but a game's last; game g's run starts at off[g] - g
- `pol_off`     (P+1,)   sparse policy offsets per position
- `pol_idx`     (E,)     our policy index of each legal move (uint16)
- `pol_p`       (E,)     its target probability (float16)
- `wdl`         (P, 3)   game result, side to move, as win/draw/loss
- `search_wdl`  (P, 3)   the search's root value at that position, same form
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

import numpy as np

import chessbot_core as core

FIELDS = ("fens", "game_off", "played", "pol_off", "pol_idx", "pol_p", "wdl", "search_wdl")


@dataclass
class Game:
    fen: str
    played: np.ndarray      # (P-1,) uint16
    pol_off: np.ndarray     # (P+1,) int64, local
    pol_idx: np.ndarray     # (E,) uint16
    pol_p: np.ndarray       # (E,) float16
    wdl: np.ndarray         # (P, 3) float16
    search_wdl: np.ndarray  # (P, 3) float16


def write_shard(path: str, games: list[Game]) -> None:
    """Write atomically: a shard directory either exists whole or not at all."""
    tmp = path + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    sizes = np.array([len(g.wdl) for g in games], np.int64)
    game_off = np.zeros(len(games) + 1, np.int64)
    np.cumsum(sizes, out=game_off[1:])
    pol_off = [np.zeros(1, np.int64)]
    base = 0
    for g in games:
        pol_off.append(g.pol_off[1:] + base)
        base += int(g.pol_off[-1])
    cat = lambda xs, dt, shape=(0,): np.concatenate(xs).astype(dt) if xs else np.zeros(shape, dt)
    arrays = {
        "fens": np.array([g.fen for g in games]),
        "game_off": game_off,
        "played": cat([g.played for g in games], np.uint16),
        "pol_off": np.concatenate(pol_off),
        "pol_idx": cat([g.pol_idx for g in games], np.uint16),
        "pol_p": cat([g.pol_p for g in games], np.float16),
        "wdl": cat([g.wdl for g in games], np.float16, (0, 3)),
        "search_wdl": cat([g.search_wdl for g in games], np.float16, (0, 3)),
    }
    for k, v in arrays.items():
        np.save(os.path.join(tmp, k + ".npy"), v)
    shutil.rmtree(path, ignore_errors=True)
    os.rename(tmp, path)


class Shard:
    def __init__(self, path: str):
        self.path = path
        for k in FIELDS:
            mode = None if k == "fens" else "r"
            setattr(self, k, np.load(os.path.join(path, k + ".npy"), mmap_mode=mode))
        self.games = len(self.fens)
        self.positions = int(self.game_off[-1])

    def game(self, g: int) -> tuple[str, np.ndarray, int, int]:
        lo, hi = int(self.game_off[g]), int(self.game_off[g + 1])
        played = np.asarray(self.played[lo - g: hi - g - 1])
        return str(self.fens[g]), played, lo, hi


def open_shards(roots: list[str], skip: set[str] | frozenset = frozenset(),
                allow_empty: bool = False) -> list[Shard]:
    """Open every shard under `roots`, except paths in `skip`.

    Each shard holds one file descriptor per memory-mapped field (Python's mmap
    keeps a dup of the fd), so a rescan must not re-open what is already open.
    """
    out = []
    for root in roots:
        if os.path.exists(os.path.join(root, "game_off.npy")):
            if root not in skip:
                out.append(Shard(root))
            continue
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if (p not in skip and not name.endswith(".tmp")
                    and os.path.exists(os.path.join(p, "game_off.npy"))):
                out.append(Shard(p))
    if not out and not allow_empty:
        raise FileNotFoundError(f"no shards under {roots}")
    return out


class Sampler:
    """A shuffle buffer of replayed positions.

    Whole games go in (consecutive positions are near-duplicates, so they are
    scattered over random slots), and each batch drawn is paid for by replaying
    about as many new positions as it took out. At the default 500k slots the
    buffer is ~1 GB and a position is drawn about once before it is replaced.

    `width` caps the stored legal moves per position. Real games almost never
    exceed it; when one does, the lowest-probability moves are dropped.
    """

    def __init__(self, shards: list[Shard], capacity: int = 500_000, width: int = 128,
                 seed: int = 0):
        self.shards = shards
        self.rng = np.random.default_rng(seed)
        self.cap, self.width = capacity, width
        c = core.Position().encode().shape[0]
        self.bits = np.zeros((capacity, c), np.uint64)
        self.scale = np.zeros((capacity, c), np.float32)
        self.idx = np.zeros((capacity, width), np.uint16)
        self.p = np.zeros((capacity, width), np.float16)
        self.wdl = np.zeros((capacity, 3), np.float16)
        self.swdl = np.zeros((capacity, 3), np.float16)
        self._reweight()
        self.filled = 0
        self.truncated = 0
        self.replayed = 0
        while self.filled < capacity:
            self._add_game(fill=True)

    def _reweight(self) -> None:
        weights = np.array([s.games for s in self.shards], np.float64)
        self.shard_p = weights / weights.sum()

    def add_shards(self, shards: list[Shard]) -> int:
        """Take on shards that appeared since the sampler was built (e.g. a
        conversion running alongside training). Returns how many were new."""
        known = {s.path for s in self.shards}
        new = [s for s in shards if s.path not in known]
        if new:
            self.shards = self.shards + new
            self._reweight()
        return len(new)

    def _add_game(self, fill: bool = False) -> int:
        s = self.shards[self.rng.choice(len(self.shards), p=self.shard_p)]
        g = int(self.rng.integers(s.games))
        fen, played, lo, hi = s.game(g)
        bits, scale = core.replay_packed(fen, played)
        n = hi - lo
        if fill:
            k = min(n, self.cap - self.filled)
            slots = np.arange(self.filled, self.filled + k)
            self.filled += k
            n = k
        else:
            slots = self.rng.integers(self.cap, size=n)
        self.bits[slots] = bits[:n]
        self.scale[slots] = scale[:n]
        self.wdl[slots] = s.wdl[lo:lo + n]
        self.swdl[slots] = s.search_wdl[lo:lo + n]
        self.idx[slots] = 0
        self.p[slots] = 0
        off = np.asarray(s.pol_off[lo:lo + n + 1])
        e_idx = np.asarray(s.pol_idx[off[0]:off[-1]])
        e_p = np.asarray(s.pol_p[off[0]:off[-1]])
        for j in range(n):
            a, b = off[j] - off[0], off[j + 1] - off[0]
            if b - a > self.width:
                keep = np.argsort(-e_p[a:b].astype(np.float32))[: self.width] + a
                self.truncated += 1
                self.idx[slots[j]] = e_idx[keep]
                self.p[slots[j]] = e_p[keep]
            else:
                self.idx[slots[j], : b - a] = e_idx[a:b]
                self.p[slots[j], : b - a] = e_p[a:b]
        self.replayed += n
        return n

    def batch(self, n: int) -> dict[str, np.ndarray]:
        rows = self.rng.integers(self.cap, size=n)
        out = {
            "bits": self.bits[rows].view(np.int64),
            "scale": self.scale[rows],
            "idx": self.idx[rows].astype(np.int64),
            "p": self.p[rows].astype(np.float32),
            "wdl": self.wdl[rows].astype(np.float32),
            "search_wdl": self.swdl[rows].astype(np.float32),
        }
        added = 0
        while added < n:
            added += self._add_game()
        return out
