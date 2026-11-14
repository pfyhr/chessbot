"""Lc0 V6 import: synthetic records from random games through python-chess.

Real lc0 data was checked separately (zero mismatches on 31,340 test80
positions, see bench/results/2026-11-12-lc0-data-as-a-warm-start.md). Random
games are here because they castle and underpromote far more often than 300
real games do, and those are the two moves the translation has to decide
from the board.
"""

import random

import chess
import numpy as np

import chessbot_core as core
from chessbot import gamestore, lc0_data

LC0 = lc0_data.lc0_moves()
LC0_INDEX = {u: i for i, u in enumerate(LC0)}
ROLES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def _rel(sq: int, black: bool) -> int:
    return chess.square_mirror(sq) if black else sq


def _lc0_uci(board: chess.Board, mv: chess.Move) -> str:
    black = board.turn == chess.BLACK
    to = mv.to_square
    if board.is_castling(mv):
        # lc0 writes castling as king-takes-rook.
        to = chess.square(7 if board.is_kingside_castling(mv) else 0, chess.square_rank(mv.from_square))
    u = chess.square_name(_rel(mv.from_square, black)) + chess.square_name(_rel(to, black))
    if mv.promotion and mv.promotion != chess.KNIGHT:
        u += chess.piece_symbol(mv.promotion)
    return u


def _record(board: chess.Board, played: chess.Move, result_q: float) -> np.ndarray:
    rec = np.zeros(1, lc0_data.V6)
    rec["version"], rec["input_format"] = 6, 1
    black = board.turn == chess.BLACK
    us, them = board.turn, not board.turn
    for i, role in enumerate(ROLES):
        for off, color in ((0, us), (6, them)):
            bb = 0
            for sq in board.pieces(role, color):
                bb |= 1 << lc0_data._lc0_bit(_rel(sq, black))
            rec["planes"][0, off + i] = bb
    probs = np.full(1858, -1.0, np.float32)
    legal = list(board.legal_moves)
    for mv in legal:
        probs[LC0_INDEX[_lc0_uci(board, mv)]] = 1.0 / len(legal)
    rec["probs"][0] = probs
    rec["played_idx"] = LC0_INDEX[_lc0_uci(board, played)]
    rec["us_oo"] = board.has_kingside_castling_rights(us)
    rec["us_ooo"] = board.has_queenside_castling_rights(us)
    rec["them_oo"] = board.has_kingside_castling_rights(them)
    rec["them_ooo"] = board.has_queenside_castling_rights(them)
    rec["stm"] = int(black)
    rec["rule50"] = board.halfmove_clock
    rec["result_q"], rec["result_d"] = result_q, 0.0
    rec["root_q"], rec["root_d"] = result_q / 2, 0.5
    return rec


def _random_game(rng: random.Random) -> tuple[bytes, list[chess.Board]]:
    board, recs, boards = chess.Board(), [], []
    while not board.is_game_over() and board.ply() < 300:
        legal = list(board.legal_moves)
        # Prefer castling and promotions so both show up in every run.
        special = [m for m in legal if board.is_castling(m) or m.promotion]
        mv = rng.choice(special) if special and rng.random() < 0.7 else rng.choice(legal)
        boards.append(board.copy())
        recs.append(_record(board, mv, 1.0 if len(recs) % 2 else -1.0))
        board.push(mv)
    return np.concatenate(recs).tobytes(), boards


def test_table_is_lc0s_1858_moves():
    assert len(LC0) == 1858 and len(set(LC0)) == 1858
    assert LC0[0] == "a1b1" and LC0[-1] == "h7h8b"


def test_random_games_translate_exactly():
    rng = random.Random(7)
    tables = lc0_data.Tables()
    seen = {"castle": 0, "underpromo": 0}
    for _ in range(40):
        raw, boards = _random_game(rng)
        game = lc0_data.convert_game(raw, tables, check=True)  # replay must match planes
        assert game is not None
        for k, board in enumerate(boards):
            pos = core.Position.from_fen(board.fen())
            want = {pos.policy_index(m.uci()) for m in board.legal_moves}
            got = set(game.pol_idx[game.pol_off[k]:game.pol_off[k + 1]].tolist())
            assert got == want, board.fen()
            seen["castle"] += any(board.is_castling(m) for m in board.legal_moves)
            seen["underpromo"] += any(m.promotion == chess.KNIGHT for m in board.legal_moves)
    assert seen["castle"] > 0 and seen["underpromo"] > 0, seen


def test_shard_round_trip_and_sampler(tmp_path):
    rng = random.Random(3)
    tables = lc0_data.Tables()
    games = [lc0_data.convert_game(_random_game(rng)[0], tables) for _ in range(6)]
    gamestore.write_shard(str(tmp_path / "s0"), games)
    shard = gamestore.open_shards([str(tmp_path)])[0]
    assert shard.games == 6 and shard.positions == sum(len(g.wdl) for g in games)
    sampler = gamestore.Sampler([shard], capacity=256, width=128, seed=0)
    b = sampler.batch(64)
    np.testing.assert_allclose(b["p"].sum(1), 1.0, atol=1e-2)
    np.testing.assert_allclose(b["wdl"].sum(1), 1.0, atol=1e-3)
    assert b["bits"].shape == (64, core.OBS_PLANES)


def test_sampler_takes_on_new_shards(tmp_path):
    rng = random.Random(5)
    tables = lc0_data.Tables()
    for k in range(2):
        games = [lc0_data.convert_game(_random_game(rng)[0], tables) for _ in range(3)]
        gamestore.write_shard(str(tmp_path / f"s{k}"), games)
    first = gamestore.open_shards([str(tmp_path / "s0")])
    sampler = gamestore.Sampler(first, capacity=128, seed=0)
    assert sampler.add_shards(gamestore.open_shards([str(tmp_path)])) == 1
    assert sampler.add_shards(gamestore.open_shards([str(tmp_path)])) == 0
    assert len(sampler.shards) == 2 and abs(sampler.shard_p.sum() - 1) < 1e-9
    sampler.batch(256)  # draws from both without error
    known = {s.path for s in sampler.shards}
    assert gamestore.open_shards([str(tmp_path)], skip=known, allow_empty=True) == []
