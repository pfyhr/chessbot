"""Tests for the human-playable Connect4 front end.

Almost all of this is presentation, but presentation has one genuine trap:
`Connect4.render()` writes from the *mover's* perspective, so the raw glyphs swap
every ply. Rendering that directly would show a player's own discs changing
identity each turn.
"""

from __future__ import annotations

import chessbot_core as cc
import pytest

from chessbot.play_connect4 import BOT_DISC, HUMAN_DISC, describe_value, render


def cells(board: cc.Connect4, human_first: bool) -> list[str]:
    """The six board rows of a render, stripped of the column header."""
    return render(board, human_first, None).split("\n")[:6]


def test_disc_identity_survives_the_perspective_flip():
    """A disc must keep its glyph for the rest of the game.

    The underlying render flips 'x' and 'o' every ply; if that leaked through,
    a player's discs would change identity each turn.
    """
    for human_first in (True, False):
        board = cc.Connect4()
        seen: list[list[str]] = []
        for col in [3, 3, 2, 4, 1, 5, 0]:
            board = board.after(col)
            seen.append(cells(board, human_first))

        # Every cell that was occupied at some point keeps its glyph afterwards.
        for earlier, later in zip(seen, seen[1:]):
            for row_a, row_b in zip(earlier, later):
                for a, b in zip(row_a.split(), row_b.split()):
                    if a in (HUMAN_DISC, BOT_DISC):
                        assert a == b, (
                            f"disc changed identity mid-game: {a} -> {b}\n"
                            f"{chr(10).join(later)}"
                        )


def test_first_move_belongs_to_whoever_moved_first():
    board = cc.Connect4().after(3)

    human_view = "".join(cells(board, human_first=True))
    assert HUMAN_DISC in human_view and BOT_DISC not in human_view

    bot_view = "".join(cells(board, human_first=False))
    assert BOT_DISC in bot_view and HUMAN_DISC not in bot_view


def test_both_players_are_distinguishable_without_colour():
    """Identity must not rest on hue alone -- piped output has no colour."""
    board = cc.Connect4.from_moves([3, 4])
    view = "".join(cells(board, human_first=True))
    assert HUMAN_DISC in view
    assert BOT_DISC in view
    assert HUMAN_DISC != BOT_DISC


def test_board_fills_from_the_bottom():
    board = cc.Connect4.from_moves([3])
    rows = cells(board, human_first=True)
    assert HUMAN_DISC in rows[-1], "first disc should sit on the bottom row"
    assert all(HUMAN_DISC not in r for r in rows[:-1])


@pytest.mark.parametrize(
    "value,expected",
    [(0.9, "winning"), (0.4, "better"), (0.0, "level"), (-0.4, "worse"), (-0.9, "losing")],
)
def test_value_descriptions_track_the_sign(value, expected):
    assert expected in describe_value(value)
