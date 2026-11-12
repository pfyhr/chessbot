"""Lc0 training chunks -> our compact game store.

Lc0 publishes its self-play as gzipped V6 records, one file per game, packed
into hourly tars (storage.lczero.org/files/training_data/). Every quirk below
was checked against our core on 31,340 test80 positions with zero mismatches;
see `bench/results/2026-11-12-lc0-data-as-a-warm-start.md`.

- Records are 8,356 bytes. Only input format 1 (classic 112 planes) is
  accepted; the canonical formats flip and transpose boards.
- Plane bits are mirrored within each byte: square = rank*8 + 7 - bit%8, in
  the side-to-move's view (ranks flipped for Black, as in ours).
- The 1858-move table is every queen-line and knight move ordered by from-
  then to-square, then 66 q/r/b promotions. A plain 7th-to-8th-rank pawn move
  is the *knight* promotion.
- Castling is king-takes-rook (`e1h1`), ours is the king's two-square move.
- Illegal moves carry policy -1, so the >= 0 entries are exactly the legal set.
- `result_q` / `root_q` are from the side to move.

Because both engines index moves in the mover's frame, the translation is a
fixed table plus two decisions taken from the board: is the king on e1
(castling) and is the mover a pawn (knight promotion).
"""

from __future__ import annotations

import argparse
import gzip
import os
import tarfile
import time
from multiprocessing import Pool

import numpy as np

import chessbot_core as core

from chessbot import gamestore

V6 = np.dtype(
    [
        ("version", "<u4"), ("input_format", "<u4"),
        ("probs", "<f4", 1858), ("planes", "<u8", 104),
        ("us_ooo", "u1"), ("us_oo", "u1"), ("them_ooo", "u1"), ("them_oo", "u1"),
        ("stm", "u1"), ("rule50", "u1"), ("invariance", "u1"), ("dummy", "u1"),
        ("root_q", "<f4"), ("best_q", "<f4"), ("root_d", "<f4"), ("best_d", "<f4"),
        ("root_m", "<f4"), ("best_m", "<f4"), ("plies_left", "<f4"),
        ("result_q", "<f4"), ("result_d", "<f4"),
        ("played_q", "<f4"), ("played_d", "<f4"), ("played_m", "<f4"),
        ("orig_q", "<f4"), ("orig_d", "<f4"), ("orig_m", "<f4"),
        ("visits", "<u4"), ("played_idx", "<u2"), ("best_idx", "<u2"),
        ("policy_kld", "<f4"), ("reserved", "<u4"),
    ]
)
assert V6.itemsize == 8356


def _sq(n: int) -> str:
    return "abcdefgh"[n % 8] + str(n // 8 + 1)


def lc0_moves() -> list[str]:
    """Lc0's 1858 policy entries as UCI strings, in the mover's frame."""
    out = []
    for f in range(64):
        fr, ff = divmod(f, 8)
        for t in range(64):
            tr, tf = divmod(t, 8)
            dr, df = tr - fr, tf - ff
            if t != f and (dr == 0 or df == 0 or abs(dr) == abs(df) or {abs(dr), abs(df)} == {1, 2}):
                out.append(_sq(f) + _sq(t))
    for ff in range(8):
        for tf in (ff - 1, ff, ff + 1):
            if 0 <= tf < 8:
                for piece in "qrb":
                    out.append(_sq(48 + ff) + _sq(56 + tf) + piece)
    assert len(out) == 1858
    return out


def _lc0_bit(square: int) -> int:
    return (square // 8) * 8 + 7 - square % 8


class Tables:
    """Lc0 index -> our policy index, with the two board-dependent cases."""

    def __init__(self):
        moves = lc0_moves()
        n = len(moves)
        self.base = np.empty(n, np.int32)
        self.castle = np.full(n, -1, np.int32)     # used when our king is on e1
        self.knight = np.full(n, -1, np.int32)     # used when the mover is a pawn
        self.from_bit = np.empty(n, np.uint64)     # from-square in lc0 bit order
        for i, u in enumerate(moves):
            self.base[i] = core.relative_move_index(u)
            sq = "abcdefgh".index(u[0]) + 8 * (int(u[1]) - 1)
            self.from_bit[i] = np.uint64(1) << np.uint64(_lc0_bit(sq))
            if u in ("e1h1", "e1a1"):
                self.castle[i] = core.relative_move_index("e1g1" if u == "e1h1" else "e1c1")
            if len(u) == 4 and u[1] == "7" and u[3] == "8" and abs(ord(u[0]) - ord(u[2])) <= 1:
                self.knight[i] = core.relative_move_index(u + "n")
        self.e1_bit = np.uint64(1) << np.uint64(_lc0_bit(4))

    def map(self, lc0_idx: np.ndarray, pawns: np.ndarray, king: np.ndarray) -> np.ndarray:
        """Translate indices; `pawns`/`king` are the mover's lc0 bitboards per entry."""
        out = self.base[lc0_idx].copy()
        castle = self.castle[lc0_idx]
        on_e1 = (king & self.e1_bit) != 0
        sel = (castle >= 0) & on_e1
        out[sel] = castle[sel]
        knight = self.knight[lc0_idx]
        sel = (knight >= 0) & ((pawns & self.from_bit[lc0_idx]) != 0)
        out[sel] = knight[sel]
        return out


def _placement(planes: np.ndarray, black: bool) -> str:
    board = [None] * 64
    for i, ch in enumerate("PNBRQKpnbrqk"):
        x = int(planes[i])
        while x:
            b = (x & -x).bit_length() - 1
            x &= x - 1
            s = (b // 8) * 8 + 7 - b % 8
            if black:
                s = (7 - s // 8) * 8 + s % 8
            board[s] = ch.swapcase() if black else ch
    rows = []
    for r in range(7, -1, -1):
        row, empty = "", 0
        for f in range(8):
            p = board[r * 8 + f]
            if p is None:
                empty += 1
                continue
            row += (str(empty) if empty else "") + p
            empty = 0
        rows.append(row + (str(empty) if empty else ""))
    return "/".join(rows)


def start_fen(rec) -> str:
    """FEN of a game's first record. En passant cannot arise before a move."""
    black = bool(rec["stm"])
    us, them = ("KQ", "kq") if not black else ("kq", "KQ")
    rights = (
        (us[0] if rec["us_oo"] else "") + (us[1] if rec["us_ooo"] else "")
        + (them[0] if rec["them_oo"] else "") + (them[1] if rec["them_ooo"] else "")
    )
    rights = "".join(sorted(rights, key="KQkq".index)) or "-"
    side = "b" if black else "w"
    return f"{_placement(rec['planes'], black)} {side} {rights} - {int(rec['rule50'])} 1"


def _wdl(q: np.ndarray, d: np.ndarray) -> np.ndarray:
    w = (1.0 + q - d) / 2.0
    l = (1.0 - q - d) / 2.0
    return np.clip(np.stack([w, d, l], axis=1), 0.0, 1.0).astype(np.float16)


_REV = np.array([int(f"{i:08b}"[::-1], 2) for i in range(256)], np.uint8)


def to_our_bits(x: np.ndarray) -> np.ndarray:
    """lc0 plane bitboards -> our bit order (square i is bit i)."""
    return _REV[np.ascontiguousarray(x).view(np.uint8)].view(np.uint64).reshape(x.shape)


def convert_game(raw: bytes, tables: Tables, check: bool = True) -> gamestore.Game | None:
    """One gzipped chunk -> a Game, or None if it is not a standard-chess game."""
    rec = np.frombuffer(raw, V6)
    if len(rec) == 0 or rec["version"][0] != 6 or rec["input_format"][0] != 1:
        return None
    fen = start_fen(rec[0])
    try:
        core.Position.from_fen(fen)
    except ValueError:
        return None  # Chess960: castling rights our parser rightly rejects

    probs = rec["probs"]
    rows, cols = np.nonzero(probs >= 0)
    pawns = rec["planes"][:, 0]
    kings = rec["planes"][:, 5]
    ours = tables.map(cols, pawns[rows], kings[rows])
    played = tables.map(rec["played_idx"].astype(np.int64), pawns, kings)

    counts = np.bincount(rows, minlength=len(rec))
    pol_off = np.zeros(len(rec) + 1, np.int64)
    np.cumsum(counts, out=pol_off[1:])
    p = probs[rows, cols]
    game = gamestore.Game(
        fen=fen,
        played=played[:-1].astype(np.uint16),
        pol_off=pol_off,
        pol_idx=ours.astype(np.uint16),
        pol_p=p.astype(np.float16),
        wdl=_wdl(rec["result_q"], rec["result_d"]),
        search_wdl=_wdl(rec["root_q"], rec["root_d"]),
    )
    if check:
        # Replaying through our core raises if any played move is illegal there,
        # and the piece planes must then agree with lc0's bit for bit.
        bits, _ = core.replay_packed(fen, game.played)
        want = to_our_bits(rec["planes"][:, :12])
        if not np.array_equal(bits[:, :12], want):
            raise ValueError("replayed boards disagree with lc0's planes")
    return game


def iter_tar(path: str):
    with tarfile.open(path) as tar:
        for m in tar:
            if m.isfile() and m.name.endswith(".gz"):
                f = tar.extractfile(m)
                try:
                    yield gzip.decompress(f.read())
                except (EOFError, OSError, gzip.BadGzipFile):
                    continue


def convert_tar(args) -> dict:
    tar_path, out_dir, check = args
    tables = Tables()
    games, stats = [], {"games": 0, "skipped": 0, "positions": 0}
    for raw in iter_tar(tar_path):
        g = convert_game(raw, tables, check)
        if g is None:
            stats["skipped"] += 1
            continue
        games.append(g)
        stats["games"] += 1
        stats["positions"] += len(g.wdl)
    name = os.path.splitext(os.path.basename(tar_path))[0]
    gamestore.write_shard(os.path.join(out_dir, name), games)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tars", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--no-check", action="store_true", help="skip replay verification")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    todo = [t for t in a.tars
            if not os.path.exists(os.path.join(a.out, os.path.splitext(os.path.basename(t))[0]))]
    t0, tot = time.time(), {"games": 0, "skipped": 0, "positions": 0}
    with Pool(a.workers) as pool:
        for i, st in enumerate(pool.imap_unordered(
                convert_tar, [(t, a.out, not a.no_check) for t in todo])):
            for k in tot:
                tot[k] += st[k]
            print(f"[{i + 1}/{len(todo)}] {tot} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
