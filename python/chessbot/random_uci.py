"""A UCI engine that plays uniformly random legal moves.

The weakest possible legal player, and therefore a fixed zero point. Every rating
in this project so far has been relative -- generation N against generation 0, or
against a random mover through our own match code. This makes the random mover a
real opponent in a real tournament runner, so our engine and an external one can
be placed on the same scale.

    random-uci

It uses chessbot_core for rules so the anchor and the engine agree about legality.
"""

from __future__ import annotations

import random
import sys

import chessbot_core as cc


def main() -> None:
    rng = random.Random(0xC0FFEE)
    pos = cc.Position()

    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        cmd, args = parts[0], parts[1:]

        if cmd == "uci":
            print("id name random-mover")
            print("id author pontus")
            print("uciok", flush=True)
        elif cmd == "isready":
            print("readyok", flush=True)
        elif cmd == "ucinewgame":
            pos = cc.Position()
        elif cmd == "position":
            if args and args[0] == "startpos":
                pos, rest = cc.Position(), args[1:]
            elif args and args[0] == "fen":
                pos, rest = cc.Position.from_fen(" ".join(args[1:7])), args[7:]
            else:
                continue
            if rest and rest[0] == "moves":
                for mv in rest[1:]:
                    pos = pos.after(mv)
        elif cmd == "go":
            moves, outcome = pos.expand()
            print(f"bestmove {rng.choice(moves) if moves else '0000'}", flush=True)
        elif cmd == "quit":
            return


if __name__ == "__main__":
    main()
