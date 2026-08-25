"""Drive the engine through UCI, the way a GUI does.

python-chess implements the *GUI* side of the protocol, so these tests are the
same conversation Arena, En Croissant or fastchess will have with the engine. If
they pass, a GUI works.

They spawn a real subprocess, which pays torch's import cost, so they are kept few
and short.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import chess
import chess.engine
import pytest

REPO = Path(__file__).resolve().parent.parent
ENGINE = REPO / ".venv" / "bin" / "chess-uci"

pytestmark = pytest.mark.skipif(
    not ENGINE.exists() or not any((REPO / "runs").glob("*/gen*.pt")),
    reason="needs an installed chess-uci and at least one trained checkpoint",
)


@pytest.fixture(scope="module")
def engine():
    eng = chess.engine.SimpleEngine.popen_uci([str(ENGINE)])
    yield eng
    eng.quit()


def test_identifies_itself_and_its_options(engine):
    assert engine.id["name"].startswith("chessbot")
    for opt in ("Generation", "Sims", "RootActions", "Device"):
        assert opt in engine.options, f"missing UCI option {opt}"


def test_plays_a_legal_move_from_the_start(engine):
    board = chess.Board()
    result = engine.play(board, chess.engine.Limit(nodes=32))
    assert result.move in board.legal_moves


def test_honours_a_node_limit(engine):
    info = engine.analyse(chess.Board(), chess.engine.Limit(nodes=64))
    assert info["nodes"] > 0
    assert "score" in info and "pv" in info


def test_honours_a_clock(engine):
    """A GUI hands the engine a clock, not a node count."""
    board = chess.Board()
    result = engine.play(board, chess.engine.Limit(white_clock=5.0, black_clock=5.0))
    assert result.move in board.legal_moves


def test_plays_a_position_given_as_moves(engine):
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3"):
        board.push(chess.Move.from_uci(uci))
    result = engine.play(board, chess.engine.Limit(nodes=32))
    assert result.move in board.legal_moves


def test_plays_a_position_given_as_fen(engine):
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    result = engine.play(board, chess.engine.Limit(nodes=32))
    assert result.move in board.legal_moves


def test_reports_mate_when_it_has_one(engine):
    """Proven results are certainties and should be scored as mates, not evaluations."""
    # Back-rank mate in one: Ra8#.
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1")
    info = engine.analyse(board, chess.engine.Limit(nodes=256))
    score = info["score"].white()
    assert score.is_mate() and score.mate() > 0, f"expected a mate score, got {score}"


def test_switching_generation_changes_the_engine(engine):
    """The Generation option is how a GUI picks which checkpoint to play.

    The range comes from the engine's own advertised option rather than from
    globbing the filesystem: the engine picks the most recently trained run, so a
    generation number that exists in some *other* run is not necessarily valid.
    """
    option = engine.options["Generation"]
    board = chess.Board()
    for gen in (option.min, option.max):
        engine.configure({"Generation": gen})
        assert engine.play(board, chess.engine.Limit(nodes=32)).move in board.legal_moves
