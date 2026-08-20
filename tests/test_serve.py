"""Tests for the browser front end's game state.

Almost all of `serve.py` is plumbing, but one part can go quietly wrong: it keeps
a `chess.Board` (for notation and the UI) and a `cc.Position` (for rules and
search) side by side. If those drift, the page shows one game while the engine
plays another.
"""

from __future__ import annotations

import chess
import pytest

torch = pytest.importorskip("torch")

from chessbot.serve import Game, NetCache  # noqa: E402


class FakeCache:
    """Stands in for NetCache so the tests need no checkpoints."""

    paths = {0: None}

    def get(self, gen):
        from chessbot.net import Net
        import chessbot_core as cc

        planes, h, w = cc.OBS_SHAPE
        return Net(planes, (h, w), cc.POLICY_LEN, 1, 8).eval()


@pytest.fixture
def game():
    return Game(FakeCache(), gen=0, sims=0, human_white=True, device="cpu")


def test_the_two_boards_stay_in_sync(game):
    """chess.Board and cc.Position must always describe the same position."""
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]:
        game.push(uci)
        assert game.board.fen().split()[0] == game.pos.fen().split()[0], uci
        assert (game.board.turn == chess.WHITE) == (game.pos.turn == "white")


def test_legal_moves_match_the_rules_core(game):
    game.push("e2e4")
    grouped = game.legal()
    flat = sorted(m for moves in grouped.values() for m in moves)
    assert flat == sorted(game.pos.legal_moves())


def test_undo_restores_the_previous_position(game):
    start = game.board.fen()
    game.push("e2e4")
    game.push("e7e5")
    game.undo()  # takes back a full move, both plies
    assert game.board.fen() == start
    assert game.pos.fen().split()[0] == start.split()[0]
    assert game.history == []


def test_undo_on_an_empty_game_is_harmless(game):
    game.undo()
    assert game.history == []


def test_illegal_moves_are_refused(game):
    with pytest.raises(ValueError):
        game.push("e2e5")
    assert game.history == []


def test_status_reports_a_finished_game(game):
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:  # Fool's mate
        game.push(uci)
    assert game.pos.outcome() == "loss"
    assert game.status() == "you lose"


def test_history_is_algebraic(game):
    game.push("e2e4")
    game.push("b8c6")
    assert game.history == ["e4", "Nc6"]


def test_state_payload_has_what_the_page_needs(game):
    s = game.state()
    for key in ("fen", "legal", "history", "status", "your_turn", "last",
                "eval", "eval_white", "human_white", "gen", "sims"):
        assert key in s, key
