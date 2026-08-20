"""Play chess against any saved generation.

The chess counterpart of `play_connect4`, and the same idea: every generation is
a checkpoint, so "how good was it after N generations?" is a question you answer
by playing it rather than by reading a number off a table.

    chess-play --gen 12                 # a specific generation
    chess-play                          # the latest checkpoint
    chess-play --ladder                 # staircase to find your level
    chess-play --gen 12 --device cpu    # leave the GPU to a training run

Moves are entered in SAN (`e4`, `Nf3`, `O-O`) or UCI (`e2e4`). python-chess does
the parsing and the algebraic notation -- it is the front end only, never in a
search loop, which is the same rule the rest of the project follows.

Fair warning: a network trained for an hour plays badly. That is the point of the
ladder.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import chess
import chessbot_core as cc
import numpy as np
import torch

from .net import Net

RESET, DIM, BOLD = "\033[0m", "\033[2m", "\033[1m"
LIGHT_SQ, DARK_SQ = "\033[48;5;180m", "\033[48;5;137m"
WHITE_PC, BLACK_PC = "\033[38;5;231m", "\033[38;5;232m"
HILITE = "\033[48;5;150m"
INFO, WARN = "\033[38;5;33m", "\033[38;5;178m"

GLYPH = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}


def colour() -> bool:
    return sys.stdout.isatty()


def paint(text: str, code: str) -> str:
    return f"{code}{text}{RESET}" if colour() else text


# --- checkpoints ------------------------------------------------------------


def find_checkpoints(run: Path) -> dict[int, Path]:
    out = {}
    for p in sorted(run.glob("gen*.pt")):
        m = re.search(r"gen(\d+)\.pt$", p.name)
        if m:
            out[int(m.group(1))] = p
    if not out:
        raise SystemExit(
            f"no gen*.pt checkpoints in {run}\n"
            f"  (the path is relative to the current directory -- "
            f"pass --run /path/to/run if you are elsewhere)"
        )
    return out


def load(path: Path, blocks: int, channels: int, device: str) -> Net:
    planes, h, w = cc.OBS_SHAPE
    net = Net(planes, (h, w), cc.POLICY_LEN, blocks, channels).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net.eval()


# --- board ------------------------------------------------------------------


def render(board: chess.Board, human_white: bool, last: chess.Move | None) -> str:
    """Draw the board from the human's side.

    Pieces are distinguished by glyph as well as colour, so the board survives a
    pipe or a monochrome terminal.
    """
    ranks = range(7, -1, -1) if human_white else range(8)
    files = range(8) if human_white else range(7, -1, -1)
    touched = {last.from_square, last.to_square} if last else set()

    lines = []
    for r in ranks:
        row = f" {r + 1} "
        for f in files:
            sq = chess.square(f, r)
            piece = board.piece_at(sq)
            glyph = GLYPH[piece.symbol()] if piece else " "
            body = f"{glyph} "
            if not colour():
                row += (piece.symbol() if piece else ".") + " "
                continue
            bg = HILITE if sq in touched else (LIGHT_SQ if (f + r) % 2 else DARK_SQ)
            fg = WHITE_PC if piece and piece.color == chess.WHITE else BLACK_PC
            row += f"{bg}{fg}{body}{RESET}"
        lines.append(row)

    header = "   " + " ".join(chess.FILE_NAMES[f] for f in files)
    return "\n".join(lines) + "\n" + paint(header, DIM)


def describe(v: float, who: str) -> str:
    if v > 0.6:
        return f"{who} thinks it is winning"
    if v > 0.2:
        return f"{who} thinks it is better"
    if v > -0.2:
        return f"{who} thinks it is level"
    if v > -0.6:
        return f"{who} thinks it is worse"
    return f"{who} thinks it is losing"


# --- engine -----------------------------------------------------------------


def engine_move(net, pos: cc.Position, device: str, sims: int, seed: int):
    """Returns (uci, value). `sims = 0` plays the raw policy."""
    if sims <= 0:
        obs = cc.encode_batch([pos])
        with torch.no_grad():
            logits, wdl = net(torch.from_numpy(obs).to(device))
            p = torch.softmax(wdl.float(), dim=-1)
            value = float(p[0, 0] - p[0, 2])
        logits = logits.float().cpu().numpy()[0]
        legal = pos.legal_moves()
        best = max(legal, key=lambda m: logits[pos.policy_index(m)])
        return best, value

    search = cc.ChessSearch([pos], sims=sims, max_considered=16, seed=seed)
    while (obs := search.next_batch()) is not None:
        with torch.no_grad():
            logits, wdl = net(torch.from_numpy(obs).to(device))
            p = torch.softmax(wdl.float(), dim=-1)
            values = (p[:, 0] - p[:, 2]).cpu().numpy()
        search.submit(
            np.ascontiguousarray(logits.float().cpu().numpy()),
            np.ascontiguousarray(values.astype(np.float32)),
        )
    return search.moves()[0], float(search.values()[0])


def prompt(board: chess.Board) -> chess.Move | None:
    """Read a move in SAN or UCI. Returns None to quit."""
    while True:
        try:
            raw = input("  your move (SAN or UCI, 'l' to list, 'q' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw.lower() in {"q", "quit", "resign"}:
            return None
        if raw.lower() in {"l", "list"}:
            print("  " + " ".join(sorted(board.san(m) for m in board.legal_moves)))
            continue
        for parse in (board.parse_san, chess.Move.from_uci):
            try:
                mv = parse(raw)
            except Exception:
                continue
            if mv in board.legal_moves:
                return mv
        print(paint(f"  '{raw}' is not a legal move here", WARN))


# --- one game ---------------------------------------------------------------


def play_game(net, device: str, sims: int, human_white: bool, seed: int, label: str):
    """Returns 'human', 'bot', 'draw', or None if abandoned."""
    board = chess.Board()
    pos = cc.Position()
    last = None

    side = "White" if human_white else "Black"
    print(f"\n  you ({side}) vs {paint(label, INFO)}   {DIM}search {sims} sims{RESET if colour() else ''}")

    ply = 0
    while True:
        print()
        print(render(board, human_white, last))
        print()

        outcome = pos.outcome()
        if outcome is not None:
            if outcome == "draw":
                print(f"  drawn — {board.result()}\n")
                return "draw"
            human_lost = (board.turn == chess.WHITE) == human_white
            print("  " + (paint("you lose", WARN) if human_lost else paint("you win", INFO)) + "\n")
            return "bot" if human_lost else "human"

        if (board.turn == chess.WHITE) == human_white:
            mv = prompt(board)
            if mv is None:
                print("\n  abandoned\n")
                return None
            san = board.san(mv)
            print(f"  you play {BOLD if colour() else ''}{san}{RESET if colour() else ''}")
        else:
            uci, value = engine_move(net, pos, device, sims, seed + ply)
            mv = chess.Move.from_uci(uci)
            san = board.san(mv)
            print(f"  {paint(label, INFO)} plays {BOLD if colour() else ''}{san}{RESET if colour() else ''}"
                  f"   {DIM}({describe(value, 'it')}){RESET if colour() else ''}")

        board.push(mv)
        pos = pos.after(mv.uci())
        last = mv
        ply += 1


# --- staircase ---------------------------------------------------------------


def ladder(checkpoints: dict[int, Path], args, device: str) -> None:
    gens = sorted(checkpoints)
    lo, hi = 0, len(gens) - 1
    idx = len(gens) // 2
    history: list[tuple[int, str]] = []
    human_white = True

    print(f"\n  {BOLD if colour() else ''}staircase{RESET if colour() else ''}: "
          f"gen{gens[0]:03d} to gen{gens[-1]:03d}")
    print("  win and you face a later generation; lose and you drop back.\n")

    while lo <= hi:
        gen = gens[idx]
        net = load(checkpoints[gen], args.blocks, args.channels, device)
        result = play_game(net, device, args.sims, human_white, args.seed + gen, f"gen{gen:03d}")
        if result is None:
            break
        history.append((gen, result))
        human_white = not human_white

        if result == "bot":
            hi = idx - 1
        else:  # a win or a draw moves you up
            lo = idx + 1
        if lo > hi:
            break
        idx = (lo + hi) // 2

    print(f"\n  {BOLD if colour() else ''}results{RESET if colour() else ''}")
    for gen, result in history:
        print(f"    gen{gen:03d}  " + {"human": "you won", "bot": "you lost", "draw": "drawn"}[result])
    lost_to = [g for g, r in history if r == "bot"]
    beaten = [g for g, r in history if r == "human"]
    print()
    if lost_to:
        print(f"  earliest generation that beat you: gen{min(lost_to):03d}")
    if beaten:
        print(f"  latest generation you beat:        gen{max(beaten):03d}")
    if not lost_to:
        print("  no generation beat you yet — try more --sims, or wait for training")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Play chess against a trained generation.")
    ap.add_argument("--run", type=Path, default=Path("runs/chess-v1"))
    ap.add_argument("--gen", type=int, help="generation to play (default: the latest)")
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--sims", type=int, default=32, help="search simulations; 0 = raw policy")
    ap.add_argument("--black", action="store_true", help="play Black")
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="use cpu to stay out of a training run's way",
    )
    args = ap.parse_args()

    checkpoints = find_checkpoints(args.run)
    if args.ladder:
        ladder(checkpoints, args, args.device)
        return

    gen = args.gen if args.gen is not None else max(checkpoints)
    if gen not in checkpoints:
        raise SystemExit(
            f"gen {gen} not found; have {min(checkpoints)}–{max(checkpoints)}"
        )
    net = load(checkpoints[gen], args.blocks, args.channels, args.device)
    play_game(net, args.device, args.sims, not args.black, args.seed, f"gen{gen:03d}")


if __name__ == "__main__":
    main()
