"""The chess gates are only worth anything if their ground truth is right.

These suites are generated with the Rust core for speed, then checked here
against python-chess, which is an independent implementation. A gate built on a
wrong answer key is worse than no gate: it reports confidently and wrongly.
"""

from __future__ import annotations

import chess
import pytest

from chessbot import chess_eval as ev


def mating_moves(board: chess.Board) -> set[str]:
    out = set()
    for m in list(board.legal_moves):
        board.push(m)
        if board.is_checkmate():
            out.add(m.uci())
        board.pop()
    return out


def safe_replies(board: chess.Board) -> set[str]:
    """Replies after which the opponent has no mate in one."""
    out = set()
    for m in list(board.legal_moves):
        board.push(m)
        allows = (not board.is_game_over()) and bool(mating_moves(board))
        if not allows:
            out.add(m.uci())
        board.pop()
    return out


@pytest.fixture(scope="module")
def tactics():
    return ev.build_tactics(mates=40, defences=40, seed=3)


def test_suite_has_both_kinds(tactics):
    kinds = {k: sum(1 for t in tactics if t.kind == k) for k in ("mate", "defend")}
    assert kinds["mate"] == 40
    assert kinds["defend"] == 40


def test_mate_answers_match_python_chess(tactics):
    for t in (x for x in tactics if x.kind == "mate"):
        truth = mating_moves(chess.Board(t.fen))
        assert truth == set(t.answers), t.fen
        assert truth, "a mate suite entry with no mate"


def test_defence_answers_match_python_chess(tactics):
    for t in (x for x in tactics if x.kind == "defend"):
        truth = safe_replies(chess.Board(t.fen))
        assert truth == set(t.answers), t.fen


def test_defence_positions_are_actually_hard(tactics):
    """Most replies must lose, or the position is passed by accident.

    Without this filter the random-move baseline on the defence suite is 0.75 and
    the metric has almost no headroom.
    """
    for t in (x for x in tactics if x.kind == "defend"):
        legal = len(t.position.legal_moves())
        assert len(t.answers) / legal <= 0.34, f"too easy: {t.fen}"
        assert t.answers, "no safe reply at all -- that is lost, not a defence"


def test_baselines_leave_room_to_improve(tactics):
    """A gate whose random baseline is already high cannot show learning."""
    base = ev.baselines(tactics)
    assert base["mate"] < 0.10, f"mate baseline too high: {base['mate']:.3f}"
    assert base["defend"] < 0.35, f"defend baseline too high: {base['defend']:.3f}"


def test_positions_are_not_terminal(tactics):
    for t in tactics:
        assert t.position.outcome() is None, f"terminal position in suite: {t.fen}"
