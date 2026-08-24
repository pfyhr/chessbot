"""A clickable board for playing the trained network.

The terminal client works, but choosing moves by typing SAN and running `l` to
remember what is legal is more bookkeeping than chess. This serves a local page
where the legal moves for the piece you touched are simply shown on the board.

    chess-serve                      # http://127.0.0.1:8000
    chess-serve --run runs/chess-v1 --device cpu

Standard library only -- no web framework -- and it binds to loopback. Game state
lives in this process; it is a single-player local tool, not a service.
"""

from __future__ import annotations

import argparse
import json
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import chess
import chessbot_core as cc
import numpy as np
import torch

from ._webui import PAGE
from .net import Net


class Game:
    """One game against one checkpoint, with the parallel positions kept in sync.

    `chess.Board` drives notation and the UI; `cc.Position` drives the rules and
    the search, and carries the repetition history that a FEN would lose.
    """

    def __init__(
        self,
        nets: "NetCache",
        gen: int,
        sims: int,
        human_white: bool,
        device: str,
        considered: int = 32,
    ):
        self.nets, self.device = nets, device
        self.gen, self.sims, self.human_white = gen, sims, human_white
        self.considered = considered
        self.reset()

    def reset(self) -> None:
        self.board = chess.Board()
        self.pos = cc.Position()
        self.history: list[str] = []
        self.stack: list[tuple] = []
        self.last: list[str] = []
        self.eval: float | None = None
        self.eval_white: bool = True
        if not self.human_white:
            self.engine_move()

    # --- rules -------------------------------------------------------------

    def legal(self) -> dict[str, list[str]]:
        """UCI moves grouped by origin square, for the board to highlight."""
        out: dict[str, list[str]] = {}
        for m in self.board.legal_moves:
            out.setdefault(chess.square_name(m.from_square), []).append(m.uci())
        return out

    def status(self) -> str | None:
        outcome = self.pos.outcome()
        if outcome is None:
            return None
        if outcome == "draw":
            return f"drawn — {self.board.result()}"
        human_lost = (self.board.turn == chess.WHITE) == self.human_white
        return "you lose" if human_lost else "you win"

    def your_turn(self) -> bool:
        return (self.board.turn == chess.WHITE) == self.human_white

    # --- moves -------------------------------------------------------------

    def push(self, uci: str) -> None:
        mv = chess.Move.from_uci(uci)
        if mv not in self.board.legal_moves:
            raise ValueError(f"illegal move {uci}")
        self.stack.append((self.board.copy(), self.pos, list(self.history), list(self.last)))
        self.history.append(self.board.san(mv))
        self.board.push(mv)
        self.pos = self.pos.after(uci)
        self.last = [uci[:2], uci[2:4]]

    def undo(self) -> None:
        """Take back a full move -- yours and the engine's reply."""
        for _ in range(2):
            if not self.stack:
                break
            self.board, self.pos, self.history, self.last = self.stack.pop()

    def engine_move(self) -> None:
        if self.pos.outcome() is not None:
            return
        net = self.nets.get(self.gen)
        uci, value = search(
            net, self.pos, self.device, self.sims, len(self.history), self.considered
        )
        self.eval, self.eval_white = value, self.board.turn == chess.WHITE
        self.push(uci)

    # --- wire format --------------------------------------------------------

    def state(self) -> dict:
        return {
            "fen": self.board.fen(),
            "legal": self.legal(),
            "history": self.history,
            "status": self.status(),
            "your_turn": self.your_turn(),
            "last": self.last,
            "eval": self.eval,
            "eval_white": self.eval_white,
            "human_white": self.human_white,
            "gen": self.gen,
            "sims": self.sims,
        }


def search(net, pos: cc.Position, device: str, sims: int, seed: int, considered: int = 32):
    """Returns (uci, value) from the moving side's point of view.

    `considered` is the root-action count. 16 was the training default and costs
    roughly 160 Elo against 32: with ~40 legal moves, the best one is often never
    sampled. Interactive play has no throughput constraint, so it uses the wider
    setting.
    """
    if sims <= 0:
        obs = cc.encode_batch([pos])
        with torch.no_grad():
            logits, wdl = net(torch.from_numpy(obs).to(device))
            p = torch.softmax(wdl.float(), dim=-1)
            value = float(p[0, 0] - p[0, 2])
        logits = logits.float().cpu().numpy()[0]
        best = max(pos.legal_moves(), key=lambda m: logits[pos.policy_index(m)])
        return best, value

    s = cc.ChessSearch([pos], sims=sims, max_considered=considered, seed=seed)
    while (obs := s.next_batch()) is not None:
        with torch.no_grad():
            logits, wdl = net(torch.from_numpy(obs).to(device))
            p = torch.softmax(wdl.float(), dim=-1)
            values = (p[:, 0] - p[:, 2]).cpu().numpy()
        s.submit(
            np.ascontiguousarray(logits.float().cpu().numpy()),
            np.ascontiguousarray(values.astype(np.float32)),
        )
    return s.moves()[0], float(s.values()[0])


class NetCache:
    """Loads checkpoints on demand and keeps them, so switching generation is instant."""

    def __init__(self, run: Path, blocks: int, channels: int, device: str):
        self.run, self.blocks, self.channels, self.device = run, blocks, channels, device
        self.cache: dict[int, Net] = {}
        self.paths = {}
        for p in sorted(run.glob("gen*.pt")):
            m = re.search(r"gen(\d+)\.pt$", p.name)
            if m:
                self.paths[int(m.group(1))] = p
        if not self.paths:
            raise SystemExit(f"no gen*.pt checkpoints in {run}")

    def get(self, gen: int) -> Net:
        if gen not in self.cache:
            planes, h, w = cc.OBS_SHAPE
            net = Net(planes, (h, w), cc.POLICY_LEN, self.blocks, self.channels).to(self.device)
            net.load_state_dict(torch.load(self.paths[gen], map_location=self.device))
            self.cache[gen] = net.eval()
        return self.cache[gen]


def make_handler(game_ref: dict, nets: NetCache, device: str, considered: int):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the console for the app, not the access log
            pass

        def _send(self, payload, code=200, ctype="application/json"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                return self._send(PAGE.encode(), ctype="text/html; charset=utf-8")
            if self.path == "/api/state":
                return self._send(game_ref["game"].state())
            if self.path == "/api/gens":
                return self._send({"gens": sorted(nets.paths)})
            self._send({"error": "not found"}, 404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            game = game_ref["game"]
            try:
                if self.path == "/api/move":
                    game.push(body["uci"])
                    if game.pos.outcome() is None:
                        game.engine_move()
                elif self.path == "/api/new":
                    game_ref["game"] = game = Game(
                        nets, body["gen"], body["sims"], body["human_white"], device,
                        considered,
                    )
                elif self.path == "/api/undo":
                    game.undo()
                elif self.path == "/api/config":
                    game.sims = body["sims"]
                else:
                    return self._send({"error": "not found"}, 404)
            except Exception as exc:  # surfaced in the page, not swallowed
                return self._send({"error": str(exc)}, 400)
            self._send(game.state())

    return Handler


def newest_run(root: Path) -> Path:
    """The most recently trained run directory.

    Defaulting to a fixed name goes stale the moment a better run finishes -- and
    quietly, since an old checkpoint still loads and still plays.
    """
    candidates = [d for d in root.glob("*") if d.is_dir() and any(d.glob("gen*.pt"))]
    if not candidates:
        raise SystemExit(f"no run directories with checkpoints under {root}")
    return max(candidates, key=lambda d: max(p.stat().st_mtime for p in d.glob("gen*.pt")))


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a clickable board.")
    ap.add_argument("--run", type=Path, default=None,
                    help="run directory; defaults to the most recently trained one")
    ap.add_argument("--gen", type=int, help="starting generation (default: the latest)")
    ap.add_argument("--sims", type=int, default=128,
                    help="higher than training: interactive play is not throughput-bound")
    ap.add_argument("--considered", type=int, default=32, help="root actions to search")
    ap.add_argument("--black", action="store_true", help="start playing Black")
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="use cpu to stay out of a training run's way",
    )
    args = ap.parse_args()

    run = args.run or newest_run(Path("runs"))
    nets = NetCache(run, args.blocks, args.channels, args.device)
    gen = args.gen if args.gen is not None else max(nets.paths)
    game_ref = {
        "game": Game(nets, gen, args.sims, not args.black, args.device, args.considered)
    }

    url = f"http://127.0.0.1:{args.port}"
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port),
        make_handler(game_ref, nets, args.device, args.considered),
    )
    print(f"chessbot on {url}")
    print(f"  {run}  ({len(nets.paths)} generations)  "
          f"{args.device}  {args.sims} sims / {args.considered} root actions")
    print("ctrl-c to stop")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
