"""Tests for the Rust/Python bridge.

The encoding itself is verified in test_encoding_vs_python_chess.py against an
independent implementation. What is checked here is that the *bridge* faithfully
exposes it: that batching agrees with encoding one at a time, that masks line up
with the policy indices, and that the rules surface Python sees matches
python-chess.
"""

from __future__ import annotations

import chess
import chessbot_core as cc
import numpy as np
import pytest


@pytest.fixture(scope="module")
def positions() -> list[cc.Position]:
    """A spread of positions from random play, including terminal ones."""
    rng = np.random.default_rng(20260816)
    out: list[cc.Position] = []
    while len(out) < 400:
        pos = cc.Position()
        for _ in range(200):
            out.append(pos)
            moves, outcome = pos.expand()
            if outcome is not None or not moves:
                break
            pos = pos.after(moves[rng.integers(len(moves))])
    return out[:400]


def test_module_constants():
    assert cc.POLICY_LEN == 4672
    assert cc.OBS_SHAPE == (119, 8, 8)
    assert cc.OBS_PLANES == 119
    assert cc.HISTORY == 8


def test_startpos_matches_python_chess():
    p = cc.Position()
    assert p.fen() == chess.Board().fen()
    assert p.turn == "white"
    assert sorted(p.legal_moves()) == sorted(m.uci() for m in chess.Board().legal_moves)


def test_legal_moves_agree_with_python_chess(positions):
    for p in positions:
        board = chess.Board(p.fen())
        assert sorted(p.legal_moves()) == sorted(board.uci(m) for m in board.legal_moves), p.fen()


def test_encode_batch_matches_individual_encodes(positions):
    """Batching must be an optimisation, not a different computation."""
    sample = positions[:64]
    batched = cc.encode_batch(sample)
    assert batched.shape == (len(sample), *cc.OBS_SHAPE)
    assert batched.dtype == np.float32
    assert batched.flags["C_CONTIGUOUS"], "torch.from_numpy wants contiguous input"

    for i, p in enumerate(sample):
        np.testing.assert_array_equal(batched[i], p.encode(), err_msg=p.fen())


def test_legal_mask_batch_matches_individual(positions):
    sample = positions[:64]
    batched = cc.legal_mask_batch(sample)
    assert batched.shape == (len(sample), cc.POLICY_LEN)
    assert batched.dtype == np.bool_

    for i, p in enumerate(sample):
        np.testing.assert_array_equal(batched[i], p.legal_mask(), err_msg=p.fen())


def test_legal_mask_marks_exactly_the_legal_moves(positions):
    for p in positions:
        moves = p.legal_moves()
        mask = p.legal_mask()
        assert int(mask.sum()) == len(moves), p.fen()
        for uci in moves:
            assert mask[p.policy_index(uci)], f"{uci} not marked at {p.fen()}"


def test_expand_agrees_with_separate_calls(positions):
    """The fused call must return exactly what the two separate ones would."""
    for p in positions:
        moves, outcome = p.expand()
        assert moves == p.legal_moves(), p.fen()
        assert outcome == p.outcome(), p.fen()


def test_terminal_positions_report_correctly():
    # Fool's mate: White to move and mated.
    mated = cc.Position.from_fen(
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    )
    assert mated.expand() == ([], "loss")

    # Stalemate: Black to move, no legal moves, not in check.
    stale = cc.Position.from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    moves, outcome = stale.expand()
    assert moves == [] and outcome == "draw"


def test_a_draw_can_be_terminal_with_moves_available():
    """The contract that is easiest to get wrong downstream.

    A fifty-move draw has legal moves and is still over, so search must check
    the outcome before it looks at the move list.
    """
    pos = cc.Position.from_fen("7k/8/6K1/8/8/8/8/R7 w - - 100 200")
    moves, outcome = pos.expand()
    assert outcome == "draw"
    assert moves, "this position does have legal moves -- that is the point"


def test_play_and_after_are_consistent():
    p = cc.Position()
    q = p.after("e2e4")
    assert p.fen() == chess.Board().fen(), "after() must not mutate the receiver"

    r = cc.Position()
    r.play("e2e4")
    assert r.fen() == q.fen()


def test_illegal_moves_raise():
    p = cc.Position()
    with pytest.raises(ValueError):
        p.play("e2e5")
    with pytest.raises(ValueError):
        p.play("not a move")


def test_repetition_is_visible_from_python():
    p = cc.Position()
    assert p.repetition_count() == 1
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]:
        p = p.after(uci)
    assert p.repetition_count() == 2
    assert p.outcome() is None

    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]:
        p = p.after(uci)
    assert p.repetition_count() == 3
    assert p.outcome() == "draw"


def test_encoding_is_torch_ready(positions):
    """The array must hand off to torch without a copy or a dtype surprise."""
    torch = pytest.importorskip("torch")
    batch = cc.encode_batch(positions[:32])
    tensor = torch.from_numpy(batch)
    assert tensor.shape == (32, *cc.OBS_SHAPE)
    assert tensor.dtype == torch.float32
