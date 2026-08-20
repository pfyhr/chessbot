"""Play Connect4 against any generation of the trained net.

This is the time-ladder idea from the plan, rehearsed at Connect4 scale: every
generation is a saved checkpoint, so "how good was it after N generations?" is a
question you can answer by playing it rather than by reading a number.

    # play one generation
    PYTHONPATH=python .venv/bin/python -m chessbot.play_connect4 --gen 39

    # find the generation that matches you, by staircase search
    PYTHONPATH=python .venv/bin/python -m chessbot.play_connect4 --ladder

The staircase is the point. Playing all forty generations would take an evening;
stepping up on a loss and down on a win finds your level in a handful of games.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import chessbot_core as cc
import numpy as np
import torch

from . import connect4_eval as ev
from .net import Net

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
YOU = "\033[38;5;33m"    # blue
BOT = "\033[38;5;209m"   # orange
WARN = "\033[38;5;178m"


def supports_colour() -> bool:
    return sys.stdout.isatty()


def paint(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}" if supports_colour() else text


# --- checkpoints ------------------------------------------------------------


def find_checkpoints(run: Path) -> dict[int, Path]:
    out = {}
    for p in sorted(run.glob("gen*.pt")):
        m = re.search(r"gen(\d+)\.pt$", p.name)
        if m:
            out[int(m.group(1))] = p
    if not out:
        raise SystemExit(f"no gen*.pt checkpoints found in {run}")
    return out


def load(path: Path, blocks: int, channels: int, device: str) -> Net:
    planes, h, w = cc.CONNECT4_OBS_SHAPE
    net = Net(planes, (h, w), cc.CONNECT4_POLICY_LEN, blocks, channels).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net.eval()


# --- board rendering --------------------------------------------------------

HUMAN_DISC = "\u25cf"  # filled
BOT_DISC = "\u25cb"    # hollow


def render(board: cc.Connect4, human_first: bool, last_move: int | None) -> str:
    """Draw the board with the two players' discs distinguished.

    Two things this has to get right:

    `Connect4.render()` writes from the *mover's* perspective, which flips every
    ply. That is correct for the network and useless for a person, so this maps
    back to fixed identities -- your discs stay yours all game.

    And the discs differ in *shape*, not only colour. Piped output, a mono
    terminal, or a colour-blind reader all lose the hue; filled versus hollow
    survives all three.
    """
    raw = board.render().strip("\n").split("\n")
    grid = [line.split() for line in raw[:6]]

    # 'x' is the side to move, 'o' the other. Resolve to first/second player.
    mover_is_first = board.plies % 2 == 0
    lines = []
    for row in grid:
        cells = []
        for ch in row:
            if ch == ".":
                cells.append(paint("·", DIM))
                continue
            is_first = (ch == "x") == mover_is_first
            human = is_first == human_first
            cells.append(
                paint(HUMAN_DISC, YOU) if human else paint(BOT_DISC, BOT)
            )
        lines.append("  " + " ".join(cells))

    header = "  " + " ".join(
        paint(str(c), BOLD if c == last_move else DIM) for c in range(7)
    )
    return "\n".join(lines) + "\n" + header


def describe_value(v: float) -> str:
    """The bot's own read on the position, from its side."""
    if v > 0.6:
        return "it thinks it is winning"
    if v > 0.2:
        return "it thinks it is better"
    if v > -0.2:
        return "it thinks it is level"
    if v > -0.6:
        return "it thinks it is worse"
    return "it thinks it is losing"


# --- one game ---------------------------------------------------------------


def bot_move(net, board, device: str, sims: int, seed: int) -> tuple[int, float]:
    if sims <= 0:
        probs = ev.policy_probs(net, [board], device)[0]
        return int(probs.argmax()), float("nan")

    search = cc.Connect4Search([board], sims=sims, max_considered=7, seed=seed)
    while (obs := search.next_batch()) is not None:
        with torch.no_grad():
            logits, wdl = net(torch.from_numpy(obs).to(device))
            p = torch.softmax(wdl.float(), dim=-1)
            values = (p[:, 0] - p[:, 2]).cpu().numpy()
        search.submit(
            np.ascontiguousarray(logits.float().cpu().numpy()),
            np.ascontiguousarray(values.astype(np.float32)),
        )
    return int(search.moves()[0]), float(search.values()[0])


def prompt_human(board: cc.Connect4) -> int | None:
    """Returns a column, or None if the player resigns."""
    legal = board.legal_moves()
    while True:
        try:
            raw = input(f"  your move {sorted(legal)} (q to quit): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw in {"q", "quit", "resign"}:
            return None
        if not raw.isdigit() or int(raw) not in legal:
            print(paint(f"  '{raw}' is not one of {sorted(legal)}", WARN))
            continue
        return int(raw)


def play_game(net, device: str, sims: int, human_first: bool, seed: int, label: str):
    """Returns 'human', 'bot', 'draw', or None if abandoned."""
    board = cc.Connect4()
    last = None
    print(
        f"\n  {paint('you ' + HUMAN_DISC, YOU)} vs {paint(label + ' ' + BOT_DISC, BOT)}"
        f"   {DIM}search {sims} sims{RESET if supports_colour() else ''}"
    )
    print(f"  you move {'first' if human_first else 'second'}\n")

    ply = 0
    while True:
        print(render(board, human_first, last))
        moves, outcome = board.expand()
        if outcome is not None:
            print()
            if outcome == "draw":
                print("  drawn — board full\n")
                return "draw"
            # Side to move has lost, so the previous mover won.
            mover_is_first = board.plies % 2 == 0
            human_lost = mover_is_first == human_first
            print("  " + (paint("you lose", BOT) if human_lost else paint("you win", YOU)) + "\n")
            return "bot" if human_lost else "human"

        human_turn = (board.plies % 2 == 0) == human_first
        if human_turn:
            col = prompt_human(board)
            if col is None:
                print("\n  abandoned\n")
                return None
        else:
            col, value = bot_move(net, board, device, sims, seed + ply)
            note = "" if value != value else f"  {DIM}({describe_value(value)}){RESET}"
            print(f"  {paint(label, BOT)} plays {col}{note}")
        board = board.after(col)
        last = col
        ply += 1
        print()


# --- staircase ---------------------------------------------------------------


def ladder(checkpoints: dict[int, Path], args, device: str) -> None:
    """Step up on a loss, down on a win, and converge on the matching generation."""
    gens = sorted(checkpoints)
    lo, hi = 0, len(gens) - 1
    idx = len(gens) // 2
    history: list[tuple[int, str]] = []

    print(f"\n  {BOLD}staircase{RESET if supports_colour() else ''}: "
          f"{len(gens)} generations, from gen{gens[0]:03d} to gen{gens[-1]:03d}")
    print("  win and you face a later generation; lose and you drop back.\n")

    human_first = True
    while lo <= hi:
        gen = gens[idx]
        net = load(checkpoints[gen], args.blocks, args.channels, device)
        result = play_game(net, device, args.sims, human_first, args.seed + gen,
                           f"gen{gen:03d}")
        if result is None:
            break
        history.append((gen, result))
        human_first = not human_first

        if result == "human":
            lo = idx + 1
        elif result == "bot":
            hi = idx - 1
        else:
            print("  a draw moves you up half a step\n")
            lo = idx + 1
        if lo > hi:
            break
        idx = (lo + hi) // 2

    print(f"\n  {BOLD}results{RESET if supports_colour() else ''}")
    for gen, result in history:
        verdict = {"human": "you won", "bot": "you lost", "draw": "drawn"}[result]
        print(f"    gen{gen:03d}  {verdict}")

    beaten = [g for g, r in history if r == "human"]
    lost_to = [g for g, r in history if r == "bot"]
    print()
    if lost_to:
        print(f"  earliest generation that beat you: gen{min(lost_to):03d}")
    if beaten:
        print(f"  latest generation you beat:        gen{max(beaten):03d}")
    if not lost_to:
        print("  no generation beat you — try more --sims, or train longer")
    if not beaten:
        print("  you did not beat any generation — try fewer --sims")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Play Connect4 against a trained generation.")
    ap.add_argument("--run", type=Path, default=Path("runs/connect4-v1"))
    ap.add_argument("--gen", type=int, help="generation to play (default: the last)")
    ap.add_argument("--ladder", action="store_true", help="staircase search for your level")
    ap.add_argument("--sims", type=int, default=32, help="search simulations; 0 = raw policy")
    ap.add_argument("--second", action="store_true", help="let the bot move first")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    checkpoints = find_checkpoints(args.run)

    if args.ladder:
        ladder(checkpoints, args, args.device)
        return

    gen = args.gen if args.gen is not None else max(checkpoints)
    if gen not in checkpoints:
        raise SystemExit(f"gen {gen} not in {args.run} (have {min(checkpoints)}–{max(checkpoints)})")

    net = load(checkpoints[gen], args.blocks, args.channels, args.device)
    tally = {"human": 0, "bot": 0, "draw": 0}
    for i in range(args.games):
        result = play_game(net, args.device, args.sims, not args.second,
                           args.seed + i * 97, f"gen{gen:03d}")
        if result is None:
            break
        tally[result] += 1

    if sum(tally.values()) > 1:
        print(f"  you {tally['human']}  gen{gen:03d} {tally['bot']}  drawn {tally['draw']}\n")


if __name__ == "__main__":
    main()
