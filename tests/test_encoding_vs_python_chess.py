"""Cross-check the Rust move encoding against an independent Python one.

The Rust unit tests prove the encoding is injective and round-trips. What they
cannot prove is that it is *AlphaZero's* encoding -- a coherently wrong plane
layout satisfies every self-consistency property while training the network
against targets that mean nothing.

So this module re-derives the encoding from the paper's description, using
python-chess for the rules, and demands the two implementations agree move for
move. It simultaneously checks that shakmaty and python-chess agree on what is
legal in the first place.

Honest limitation: both implementations were written by the same author from the
same reading of the spec, so a *misreading* of the paper would survive. What this
does catch is the far likelier failure -- transcription slips, off-by-ones in the
plane arithmetic, sign errors under board mirroring, and the castling and
promotion special cases.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import chess
import pytest

REPO = Path(__file__).resolve().parent.parent
DUMP_BIN = REPO / "target" / "release" / "dump"

# --- the encoding, re-derived from the AlphaZero paper ----------------------
#
# 73 planes attached to the origin square:
#   0..56   queen moves     8 directions x 7 distances
#   56..64  knight moves    8 deltas
#   64..73  underpromotions 3 pieces x 3 directions
# Flat index is plane * 64 + origin, with the board mirrored when Black moves.

DIRECTIONS = [
    (0, 1),    # N
    (1, 1),    # NE
    (1, 0),    # E
    (1, -1),   # SE
    (0, -1),   # S
    (-1, -1),  # SW
    (-1, 0),   # W
    (-1, 1),   # NW
]

KNIGHT_DELTAS = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
]

UNDERPROMOTION_PIECES = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}

POLICY_LEN = 73 * 64


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def policy_index(board: chess.Board, move: chess.Move) -> int:
    """Independent implementation of the AlphaZero move encoding."""
    flip = board.turn == chess.BLACK
    frm = chess.square_mirror(move.from_square) if flip else move.from_square
    to = chess.square_mirror(move.to_square) if flip else move.to_square

    df = chess.square_file(to) - chess.square_file(frm)
    dr = chess.square_rank(to) - chess.square_rank(frm)

    if move.promotion is not None and move.promotion != chess.QUEEN:
        # Underpromotions get dedicated planes; queen promotions ride the pawn
        # move's own plane and fall through below.
        piece = UNDERPROMOTION_PIECES[move.promotion]
        plane = 64 + piece * 3 + (df + 1)
    elif (df, dr) in KNIGHT_DELTAS:
        plane = 56 + KNIGHT_DELTAS.index((df, dr))
    else:
        distance = max(abs(df), abs(dr))
        direction = DIRECTIONS.index((_sign(df), _sign(dr)))
        plane = direction * 7 + distance - 1

    return plane * 64 + frm


# --- fixtures ---------------------------------------------------------------


@pytest.fixture(scope="session")
def dumped_positions() -> list[tuple[str, dict[str, int]]]:
    """Run the Rust dump binary and parse its TSV output."""
    if not DUMP_BIN.exists():
        pytest.fail(
            f"{DUMP_BIN} not found -- build it first with:\n"
            f"    cargo build --release --bin dump"
        )

    result = subprocess.run(
        [str(DUMP_BIN), "--games", "400", "--max-plies", "160"],
        capture_output=True,
        text=True,
        check=True,
    )

    positions = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fen, encoded = line.split("\t")
        moves = {}
        for item in encoded.split(","):
            uci, index = item.rsplit(":", 1)
            moves[uci] = int(index)
        positions.append((fen, moves))

    assert len(positions) > 5000, f"dump was too small to mean much: {len(positions)}"
    return positions


# --- tests ------------------------------------------------------------------


def test_legal_moves_agree_with_python_chess(dumped_positions):
    """shakmaty and python-chess must generate identical legal move sets."""
    mismatches = []
    for fen, rust_moves in dumped_positions:
        board = chess.Board(fen)
        theirs = {board.uci(m) for m in board.legal_moves}
        ours = set(rust_moves)
        if theirs != ours:
            mismatches.append((fen, sorted(ours - theirs), sorted(theirs - ours)))
            if len(mismatches) >= 5:
                break

    assert not mismatches, "legal move disagreement:\n" + "\n".join(
        f"  {fen}\n    rust-only:   {only_ours}\n    python-only: {only_theirs}"
        for fen, only_ours, only_theirs in mismatches
    )


def test_policy_indices_agree_with_independent_encoder(dumped_positions):
    """Every move must encode identically in both implementations."""
    mismatches = []
    checked = 0
    for fen, rust_moves in dumped_positions:
        board = chess.Board(fen)
        for move in board.legal_moves:
            uci = board.uci(move)
            if uci not in rust_moves:
                continue  # covered by the legality test above
            expected = policy_index(board, move)
            actual = rust_moves[uci]
            checked += 1
            if expected != actual:
                mismatches.append((fen, uci, expected, actual))
                if len(mismatches) >= 10:
                    break
        if len(mismatches) >= 10:
            break

    assert checked > 100_000, f"only checked {checked} moves"
    assert not mismatches, "encoding disagreement:\n" + "\n".join(
        f"  {fen}\n    {uci}: python={exp} (plane {exp // 64}, from {exp % 64}) "
        f"rust={act} (plane {act // 64}, from {act % 64})"
        for fen, uci, exp, act in mismatches
    )


def test_indices_stay_in_range(dumped_positions):
    for fen, rust_moves in dumped_positions:
        for uci, index in rust_moves.items():
            assert 0 <= index < POLICY_LEN, f"{fen} {uci} -> {index}"


def test_indices_are_unique_within_a_position(dumped_positions):
    """The property the training targets depend on."""
    for fen, rust_moves in dumped_positions:
        indices = list(rust_moves.values())
        assert len(indices) == len(set(indices)), f"collision at {fen}"


def test_underpromotions_are_distinguishable():
    """All four promotion pieces, in all three directions, must differ."""
    # Pawn on b7 with b8 *empty* and enemy pieces on a8 and c8, so all three
    # promotion directions are available: capture left, straight, capture right.
    board = chess.Board("r1b1k3/1P6/8/8/8/8/8/4K3 w q - 0 1")
    promos = [m for m in board.legal_moves if m.promotion is not None]
    assert len(promos) == 12, f"expected 3 directions x 4 pieces, got {len(promos)}"

    # All three directions must really be present, or the test proves less than
    # it looks like it does.
    assert {chess.square_file(m.to_square) for m in promos} == {0, 1, 2}

    indices = {policy_index(board, m) for m in promos}
    assert len(indices) == 12, "promotion encodings collided"

    for m in promos:
        plane = policy_index(board, m) // 64
        if m.promotion == chess.QUEEN:
            assert plane < 56, f"queen promo on plane {plane}, expected a queen plane"
        else:
            assert 64 <= plane < 73, f"underpromo on plane {plane}"


def test_castling_is_a_two_square_king_move():
    board = chess.Board(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    )
    castles = [m for m in board.legal_moves if board.is_castling(m)]
    assert len(castles) == 2

    for m in castles:
        plane = policy_index(board, m) // 64
        # East or west, distance two: direction 2 or 6, distance index 1.
        expected = 2 * 7 + 1 if chess.square_file(m.to_square) > 4 else 6 * 7 + 1
        assert plane == expected, f"{board.uci(m)} on plane {plane}"


def test_black_encoding_mirrors_white():
    """The same position with colours and ranks swapped must encode alike."""
    white = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    black = chess.Board("4k3/4p3/8/8/8/8/8/4K3 b - - 0 1")

    wm = chess.Move.from_uci("e2e4")
    bm = chess.Move.from_uci("e7e5")
    assert wm in white.legal_moves and bm in black.legal_moves

    assert policy_index(white, wm) == policy_index(black, bm)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
