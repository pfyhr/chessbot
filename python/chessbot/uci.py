"""A UCI engine, so any real chess GUI can drive the network.

UCI is the interface every chess GUI, tournament runner and bot bridge speaks.
Implementing it once buys Arena, Cute Chess, En Croissant, Nibbler, BanksiaGUI,
`fastchess` for SPRT, and lichess-bot -- none of which need to know anything about
this project.

    chess-uci                       # speaks UCI on stdin/stdout
    chess-uci --run runs/pw-32      # a specific run

The engine is Python because the network is: search lives in Rust, but inference
is PyTorch. A GUI spawns this as a subprocess and neither knows the difference,
beyond a second or two of import time at startup.

Options exposed to the GUI:

    Generation   which checkpoint to play (default: the latest)
    Sims         search simulations per move
    RootActions  root actions considered; 16 was the training default and is
                 worth about 160 Elo less than 32
    Device       mps / cpu / cuda
"""

from __future__ import annotations

import argparse
import math
import os
import re
import select
import sys
import time
from pathlib import Path

import chessbot_core as cc
import numpy as np
import torch

from .net import Net

ENGINE_NAME = "chessbot"
ENGINE_AUTHOR = "pontus"

# Lc0's win-probability to centipawn mapping. Any monotonic transform would do;
# this one is the convention GUIs are used to reading.
def value_to_cp(v: float) -> int:
    v = max(-0.9999, min(0.9999, v))
    return int(round(111.7 * math.tan(1.5620688421 * v)))


def score_of(proven: str | None, value: float) -> str:
    """A proven result is a certainty and is reported as mate, not as a number.

    The distance to mate is not tracked, so it is stated as the shortest it could
    be: honest about the sign, approximate in depth.
    """
    if proven == "win":
        return "mate 1"
    if proven == "loss":
        return "mate -1"
    return f"cp {value_to_cp(value)}"


def wdl_of(value: float) -> str:
    """Win/draw/loss in UCI's per-mille form.

    The network produces WDL natively; centipawns are the derived quantity, so
    reporting both loses nothing and GUIs that understand WDL show the better one.
    """
    w = max(0.0, min(1.0, (value + 1) / 2))
    draw = max(0.0, 1.0 - abs(value))
    win = max(0.0, w - draw / 2)
    loss = max(0.0, 1.0 - win - draw)
    total = win + draw + loss or 1.0
    return f"wdl {int(win / total * 1000)} {int(draw / total * 1000)} {int(loss / total * 1000)}"


# Set CHESSBOT_UCI_LOG to a path to record the conversation. Protocol bugs with a
# GUI are otherwise almost impossible to see: both sides look fine in isolation.
LOG = os.environ.get("CHESSBOT_UCI_LOG")


def log(direction: str, line: str) -> None:
    if not LOG:
        return
    with open(LOG, "a") as fh:
        fh.write(f"{time.time():.3f} {direction} {line}\n")


def out(line: str) -> None:
    log(">", line)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


class Engine:
    def __init__(self, run: Path, blocks: int, channels: int, device: str):
        self.run, self.blocks, self.channels = run, blocks, channels
        self.device = device
        self.paths = {}
        for p in sorted(run.glob("gen*.pt")):
            m = re.search(r"gen(\d+)\.pt$", p.name)
            if m:
                self.paths[int(m.group(1))] = p
        if not self.paths:
            raise SystemExit(f"no gen*.pt checkpoints in {run}")

        self.generation = max(self.paths)
        self.sims = 128
        self.root_actions = 32
        self.multipv = 1
        # Commands that arrived mid-search and are not ours to act on now. They
        # must be replayed, not dropped: a GUI sends `stop`, `position` and `go`
        # back to back, and swallowing the last two desyncs it permanently.
        self.queued: list[str] = []
        self.cache: dict[int, Net] = {}
        self.pos = cc.Position()
        # Rolling estimate of simulations per second, for time controls.
        self.sps = 400.0

    # --- network ------------------------------------------------------------

    def net(self) -> Net:
        if self.generation not in self.cache:
            planes, h, w = cc.OBS_SHAPE
            n = Net(planes, (h, w), cc.POLICY_LEN, self.blocks, self.channels).to(self.device)
            n.load_state_dict(torch.load(self.paths[self.generation], map_location=self.device))
            self.cache[self.generation] = n.eval()
        return self.cache[self.generation]

    # --- protocol -----------------------------------------------------------

    def handle(self, line: str) -> bool:
        """Returns False when the GUI asks the engine to quit."""
        log("<", line)
        parts = line.split()
        if not parts:
            return True
        cmd, args = parts[0], parts[1:]

        if cmd == "uci":
            out(f"id name {ENGINE_NAME} gen{self.generation:03d}")
            out(f"id author {ENGINE_AUTHOR}")
            lo, hi = min(self.paths), max(self.paths)
            out(f"option name Generation type spin default {self.generation} min {lo} max {hi}")
            out(f"option name Sims type spin default {self.sims} min 1 max 100000")
            out(f"option name RootActions type spin default {self.root_actions} min 2 max 256")
            out("option name Device type string default " + self.device)
            out(f"option name MultiPV type spin default {self.multipv} min 1 max 256")
            out("option name UCI_ShowWDL type check default true")
            out("uciok")
        elif cmd == "isready":
            self.net()  # pay the load cost here, where the GUI expects to wait
            out("readyok")
        elif cmd == "setoption":
            self.set_option(args)
        elif cmd == "ucinewgame":
            self.pos = cc.Position()
        elif cmd == "position":
            self.set_position(args)
        elif cmd == "go":
            self.go(args)
        elif cmd == "quit":
            return False
        # "stop" and "ponderhit" are accepted and ignored: search is not
        # interruptible mid-way, and every search here is short.
        return True

    def set_option(self, args: list[str]) -> None:
        if "name" not in args:
            return
        text = " ".join(args)
        m = re.match(r"name\s+(.+?)\s+value\s+(.+)", text)
        if not m:
            return
        name, value = m.group(1).strip(), m.group(2).strip()
        if name == "Generation" and int(value) in self.paths:
            self.generation = int(value)
        elif name == "Sims":
            self.sims = max(1, int(value))
        elif name == "RootActions":
            self.root_actions = max(2, int(value))
        elif name == "Device":
            self.device, self.cache = value, {}
        elif name == "MultiPV":
            self.multipv = max(1, int(value))
        # UCI_ShowWDL is accepted; WDL is always reported, since it is what the
        # network natively produces and centipawns are the derived quantity.

    def set_position(self, args: list[str]) -> None:
        if not args:
            return
        if args[0] == "startpos":
            self.pos = cc.Position()
            rest = args[1:]
        elif args[0] == "fen":
            fen = " ".join(args[1:7])
            self.pos = cc.Position.from_fen(fen)
            rest = args[7:]
        else:
            return
        if rest and rest[0] == "moves":
            for mv in rest[1:]:
                self.pos = self.pos.after(mv)

    # --- search -------------------------------------------------------------

    def budget(self, args: list[str]) -> int:
        """Simulations to spend, from whatever the GUI asked for."""
        opts = {}
        for i, a in enumerate(args):
            if a in ("wtime", "btime", "winc", "binc", "movetime", "nodes", "depth", "movestogo"):
                if i + 1 < len(args):
                    opts[a] = int(args[i + 1])

        if "nodes" in opts:
            return max(1, opts["nodes"])
        seconds = None
        if "movetime" in opts:
            seconds = opts["movetime"] / 1000.0
        else:
            ours = opts.get("wtime" if self.pos.turn == "white" else "btime")
            if ours is not None:
                inc = opts.get("winc" if self.pos.turn == "white" else "binc", 0)
                moves_left = max(10, opts.get("movestogo", 30))
                seconds = (ours / moves_left + inc) / 1000.0
        if seconds is None:
            return self.sims
        # Leave headroom: a GUI that flags the engine is worse than one move
        # searched a little less deeply.
        return max(8, min(int(seconds * 0.7 * self.sps), 20000))

    def pending_input(self) -> list[str]:
        """Whatever the GUI has said since we last looked, without blocking."""
        lines = []
        while select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if not line:
                break
            lines.append(line.strip())
        return lines

    def go_infinite(self) -> None:
        """Analyse until told to stop, reporting as we go.

        This is the mode analysis GUIs actually use -- Nibbler and En Croissant
        send `go infinite` and expect a live stream of `info` until `stop`. A
        single burst followed by `bestmove` looks to them like an engine that
        gave up immediately.

        Deepening by doubling the simulation budget gives a stream of
        progressively better answers while staying interruptible between rounds.
        """
        best = None
        sims = 64
        total_nodes = 0
        total_time = 0.0
        quitting = False

        while True:
            best, detail, elapsed, done, nodes = self.run_search(sims)
            # `nodes` in UCI means total nodes searched, and a GUI enforcing its
            # own node limit counts them. Reporting per-move visits here left
            # Nibbler's counter crawling and its limit never triggering.
            total_nodes += nodes
            total_time += elapsed
            self.report(detail, total_time, done, total_nodes, best)

            stop = False
            for line in self.pending_input():
                if line == "stop":
                    stop = True
                elif line == "quit":
                    stop, quitting = True, True
                elif line == "isready":
                    out("readyok")
                elif line:
                    # Anything else -- `position`, `go`, `setoption` -- belongs to
                    # the next turn. Keep it and stop searching so it gets one.
                    self.queued.append(line)
                    stop = True
            if stop:
                break
            # Cap the round size: each round is a fresh search, so a huge one
            # buys little and makes `stop` wait for it. 1024 keeps the worst-case
            # response under about five seconds.
            sims = min(sims * 2, 1024)

        out(f"bestmove {best}")
        if quitting:
            raise SystemExit(0)

    def run_search(self, sims: int):
        net = self.net()
        started = time.perf_counter()
        search = cc.ChessSearch([self.pos], sims=sims, max_considered=self.root_actions, seed=0)
        while (obs := search.next_batch()) is not None:
            with torch.no_grad():
                logits, wdl = net(torch.from_numpy(obs).to(self.device))
                p = torch.softmax(wdl.float(), dim=-1)
                values = (p[:, 0] - p[:, 2]).cpu().numpy()
            search.submit(
                np.ascontiguousarray(logits.float().cpu().numpy()),
                np.ascontiguousarray(values.astype(np.float32)),
            )
        elapsed = max(1e-6, time.perf_counter() - started)
        done = int(search.simulations()[0])
        self.sps = 0.7 * self.sps + 0.3 * (done / elapsed)
        return (
            search.moves()[0],
            search.root_detail(),
            elapsed,
            done,
            int(search.nodes()[0]),
        )

    def report(self, detail, elapsed, done, nodes, best) -> None:
        """Emit one round of analysis.

        `nodes` is the cumulative total, which is what UCI means by the word and
        what a GUI counts against its own limits. Per-move counts belong in the
        `info string` lines instead.
        """
        ms, nps = int(elapsed * 1000), int(nodes / max(elapsed, 1e-6))

        for rank, (uci, prob, prior, visits, q, proven) in enumerate(detail[: self.multipv], 1):
            v = q if q is not None else 0.0
            # `pv` must be last: UCI defines both `pv` and `string` as
            # "everything after this token", so putting `string` first swallows
            # the principal variation.
            out(
                f"info multipv {rank} depth 1 seldepth 1 nodes {nodes} "
                f"time {ms} nps {nps} score {score_of(proven, v)} "
                f"{wdl_of(v)} pv {uci}"
            )

        # Per-move detail in Lc0's `info string` shape, which analysis GUIs parse
        # to show the policy prior and visit share beside each move. `P` is the
        # network's raw prior, deliberately: the gap between it and where the
        # visits went is exactly how much the search disagreed with the network,
        # and that gap is the most interesting thing on the screen.
        for uci, prob, prior, visits, q, proven in detail:
            v = q if q is not None else 0.0
            draw = max(0.0, 1.0 - abs(v))
            out(
                f"info string {uci} N: {visits} (P: {prior * 100:.2f}%) "
                f"(Q: {v:.5f}) (D: {draw:.3f}) (V: {v:.4f})"
            )
        out(f"info string node N: {nodes}")

        if not detail:
            out(f"info depth 1 nodes {nodes} time {ms} nps {nps} pv {best}")

    def go(self, args: list[str]) -> None:
        moves, outcome = self.pos.expand()
        if outcome is not None or not moves:
            out("bestmove 0000")
            return

        if "infinite" in args or "ponder" in args:
            self.go_infinite()
            return

        sims = self.budget(args)
        best, detail, elapsed, done, nodes = self.run_search(sims)
        self.report(detail, elapsed, done, nodes, best)
        out(f"bestmove {best}")


def commands(engine: "Engine"):
    """Commands to act on: anything deferred from a search first, then stdin."""
    while True:
        while engine.queued:
            yield engine.queued.pop(0)
        line = sys.stdin.readline()
        if not line:
            return
        yield line.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="UCI engine over the trained network.")
    ap.add_argument("--run", type=Path, default=None,
                    help="run directory; defaults to the most recently trained one")
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--device",
                    default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    from .serve import newest_run

    run = args.run or newest_run(Path(__file__).resolve().parents[2] / "runs")
    engine = Engine(run, args.blocks, args.channels, args.device)

    for line in commands(engine):
        if not engine.handle(line):
            break


if __name__ == "__main__":
    main()
