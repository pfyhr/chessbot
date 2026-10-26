"""Build the parity reference, and answer the fp16 question while we are here.

The browser reimplements nothing -- the search and the encoding are the same
Rust compiled to a different target -- but "the same source" is not the same as
"the same behaviour", and the ways this can go wrong are all silent:

* an encoding that transposes two planes still trains, still plays, still looks
  like chess, and is simply a different network;
* a policy index off by one plane maps every move to a neighbour's logit;
* fp16 rounding that reorders two near-tied root moves.

None of those raise an error anywhere. So this writes down exactly what the
trainer's own code produces for a fixed suite of positions, in a form the
browser can be held to: `web/tools/parity_check.mjs` recomputes all of it
through the wasm build and ONNX Runtime and asserts a match.

Two kinds of claim, kept apart on purpose:

* **Exact.** FEN, legal move list and order, policy indices, and a hash of the
  raw observation bytes. Same code, same target semantics -- these must be
  identical, and a tolerance here would only hide a real bug.
* **Within tolerance.** Network outputs, because ONNX Runtime is a different
  implementation of the same arithmetic and fp16 is a different arithmetic
  altogether. Judged on what actually matters: whether the top legal move and
  the search's chosen move still agree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import chessbot_core as cc
from chessbot.net import net_from_checkpoint

PLANES, BOARD, POLICY = 119, (8, 8), 4672


def fnv1a(buf: bytes) -> str:
    """64-bit FNV-1a over the raw bytes. Trivially reimplementable in JS."""
    h = 0xCBF29CE484222325
    for b in buf:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def random_games(n_cases: int, seed: int) -> list[list[str]]:
    """Move sequences reaching a spread of positions, history included.

    Random play rather than the net's own moves: it is reproducible from a seed
    alone, and it reaches the awkward corners -- promotions, en passant, bare
    kings, repetitions -- far faster than good play does, which is exactly where
    an encoding bug would hide.
    """
    rng = np.random.default_rng(seed)
    cases: list[list[str]] = []
    while len(cases) < n_cases:
        pos, moves = cc.Position(), []
        depth = int(rng.integers(0, 80))
        for _ in range(depth):
            legal, outcome = pos.expand()
            if outcome is not None or not legal:
                break
            mv = legal[int(rng.integers(len(legal)))]
            pos.play(mv)
            moves.append(mv)
        legal, outcome = pos.expand()
        if outcome is None and legal:
            cases.append(moves)
    return cases


def handwritten() -> list[list[str]]:
    """Positions chosen for the encoding corners rather than for variety."""
    return [
        [],  # the start position: every plane in its initial state
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],  # a normal opening
        ["e2e4", "d7d5", "e4d5", "d8d5"],  # a capture, queen out early
        ["e2e4", "e7e6", "e4e5", "d7d5"],  # en passant available
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1"],  # white castles
        # Repetition: knights out and back, twice, so the repetition planes fire.
        ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"],
        # A pawn one square from promoting, reached legally.
        ["a2a4", "b7b5", "a4b5", "a7a6", "b5a6", "b8c6", "a6a7"],
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--onnx", type=Path, action="append", required=True,
                    help="a model to check against torch; repeatable")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--selfcheck", type=Path,
                    help="where to write the page's own load-time check")
    ap.add_argument("--cases", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--search-cases", type=int, default=12)
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--considered", type=int, default=16)
    args = ap.parse_args()

    import onnxruntime as ort

    net, arch = net_from_checkpoint(args.checkpoint, PLANES, BOARD, POLICY, "cpu")
    net.eval()

    suite = handwritten() + random_games(args.cases, args.seed)
    print(f"{len(suite)} positions")

    # --- the exact half, straight from the trainer's own core ---------------
    cases = []
    positions = []
    for i, moves in enumerate(suite):
        pos = cc.Position()
        for m in moves:
            pos.play(m)
        positions.append(pos)
        obs = cc.encode_batch([pos])[0].astype(np.float32)
        legal = pos.legal_moves()
        cases.append({
            "id": i,
            "moves": moves,
            "fen": pos.fen(),
            "legal": legal,
            "idx": [pos.policy_index(m) for m in legal],
            "obs_hash": fnv1a(np.ascontiguousarray(obs).tobytes()),
            "obs_sum": round(float(obs.sum()), 4),
        })

    # --- the network, in torch, as the thing everything else is measured to --
    obs = cc.encode_batch(positions).astype(np.float32)
    with torch.no_grad():
        p, wdl = net(torch.from_numpy(obs))
        probs = torch.softmax(wdl.float(), dim=-1)
        ref_logits = p.float().numpy()
        ref_wdl = probs.numpy()
        ref_value = (probs[:, 0] - probs[:, 2]).numpy()

    for c, lg, v, w in zip(cases, ref_logits, ref_value, ref_wdl):
        c["logits"] = [round(float(lg[j]), 5) for j in c["idx"]]
        c["value"] = round(float(v), 5)
        c["wdl"] = [round(float(x), 5) for x in w]
        c["best_raw"] = c["legal"][int(np.argmax([lg[j] for j in c["idx"]]))]

    # --- every candidate model against torch -------------------------------
    report = {}
    for path in args.onnx:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        o_p, o_w, o_v = sess.run(None, {"obs": obs})
        d_logits = np.abs(o_p - ref_logits).max()
        d_value = np.abs(o_v - ref_value).max()
        # Agreement on the move, not on the floats: a 1e-3 logit shift that
        # never changes a ranking costs nothing, and one that flips the top
        # move costs everything.
        agree = 0
        kl = []
        for k, c in enumerate(cases):
            idx = c["idx"]
            a = np.array([ref_logits[k][j] for j in idx])
            b = np.array([o_p[k][j] for j in idx])
            agree += int(np.argmax(a) == np.argmax(b))
            pa = np.exp(a - a.max()); pa /= pa.sum()
            pb = np.exp(b - b.max()); pb /= pb.sum()
            kl.append(float((pa * np.log(pa / np.clip(pb, 1e-12, None))).sum()))
        report[path.name + ":" + path.parent.name] = {
            "max_abs_logit": float(d_logits),
            "max_abs_value": float(d_value),
            "top1_agree": f"{agree}/{len(cases)}",
            "kl_mean": float(np.mean(kl)),
            "kl_max": float(np.max(kl)),
        }

    # --- fixed-seed searches, driven by the model the page will ship --------
    # Run against the LAST --onnx given, which is the one that goes to the page.
    ship = args.onnx[-1]
    sess = ort.InferenceSession(str(ship), providers=["CPUExecutionProvider"])
    # Terminal roots are deliberately in the suite -- the threefold case is the
    # only one that fires both repetition planes -- but a search cannot run from
    # one, so the search cases are drawn from the positions that still have a
    # game left in them.
    playable = [i for i, p in enumerate(positions) if p.expand()[1] is None]
    searches = []
    for i in playable[: args.search_cases]:
        pos = positions[i]
        s = cc.ChessSearch([pos], sims=args.sims, max_considered=args.considered, seed=i + 1)
        while (batch := s.next_batch()) is not None:
            lg, _, val = sess.run(None, {"obs": batch.astype(np.float32)})
            s.submit(np.ascontiguousarray(lg.astype(np.float32)),
                     np.ascontiguousarray(val.astype(np.float32)))
        searches.append({
            "id": i,
            "moves": suite[i],
            "sims": args.sims,
            "considered": args.considered,
            "seed": i + 1,
            "best": s.moves()[0],
            "value": round(float(s.values()[0]), 5),
            "sims_done": int(s.simulations()[0]),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "arch": arch,
        "ship_model": ship.name,
        "planes": PLANES, "policy_len": POLICY,
        "cases": cases,
        "searches": searches,
    }) + "\n")

    # --- the page's load-time check ----------------------------------------
    # Measured against the SHIPPED model under ONNX Runtime on CPU, not against
    # torch: the page runs that exact model, so the only thing left to vary is
    # the backend. Comparing to torch instead would need a tolerance wide
    # enough to swallow fp16 rounding, and a tolerance that wide would not
    # notice a WebGPU kernel quietly computing the wrong thing.
    if args.selfcheck:
        c = cases[0]
        lg, _, val = sess.run(None, {"obs": obs[0:1]})
        args.selfcheck.parent.mkdir(parents=True, exist_ok=True)
        args.selfcheck.write_text(json.dumps({
            "note": "expected outputs of the shipped model, from ONNX Runtime on CPU",
            "moves": c["moves"],
            "fen": c["fen"],
            "legal": c["legal"],
            "logits": [round(float(lg[0][j]), 5) for j in c["idx"]],
            "value": round(float(val[0]), 5),
            "tolerance": 0.05,
        }) + "\n")
        print(f"selfcheck -> {args.selfcheck}")

    print(f"\nreference -> {args.out}  ({args.out.stat().st_size/1e6:.2f} MB)")
    print("\nONNX vs torch:")
    for name, r in report.items():
        print(f"  {name:28s} logit {r['max_abs_logit']:.5f}  value {r['max_abs_value']:.5f}"
              f"  top1 {r['top1_agree']}  KL mean {r['kl_mean']:.2e} max {r['kl_max']:.2e}")


if __name__ == "__main__":
    main()
