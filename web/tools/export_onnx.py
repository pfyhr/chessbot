"""Export a training checkpoint to ONNX for the browser build.

Three things this does that a bare `torch.onnx.export` call would not:

1. **`net.eval()`.** The trunk is full of BatchNorm. Exported in training mode
   the graph uses batch statistics, and with the batch size of 1 the browser
   actually uses, every activation is normalised against itself -- the network
   plays like noise while loading, running and looking entirely healthy.

2. **Folds the WDL head down to the scalar the search backs up.** MCTS wants
   `P(win) - P(loss)`; leaving that arithmetic in JavaScript adds a second place
   for the two paths to disagree about something neither is really responsible
   for. The probabilities come out too, because a person watching wants to see
   the draw share.

3. **Writes a manifest** next to the model recording the checkpoint it came
   from, the architecture, and the input shape, so a page that loads a stale
   model can say so rather than producing plausible-looking rubbish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from chessbot.net import load_checkpoint, net_from_checkpoint

from _provenance import provenance

PLANES, BOARD, POLICY = 119, (8, 8), 4672


class Exported(nn.Module):
    """The network plus the value reduction, as one graph."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor):
        policy, wdl = self.net(x)
        probs = F.softmax(wdl, dim=-1)
        return policy, probs, probs[:, 0] - probs[:, 2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="path for the .onnx")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument(
        "--fp16",
        action="store_true",
        help="halve the download; validated by tools/parity.py, not assumed",
    )
    args = ap.parse_args()

    net, arch = net_from_checkpoint(args.checkpoint, PLANES, BOARD, POLICY, "cpu")
    net.eval()
    model = Exported(net).eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, PLANES, *BOARD)
    # Export beside the destination, then consolidate. Torch writes tensors over
    # a size threshold to a `.onnx.data` sidecar, and ONNX Runtime Web will not
    # go looking for one -- it loads a model from a single ArrayBuffer. A model
    # that silently arrives with no weights is the failure mode to avoid here.
    staging = args.out.parent / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    raw = staging / args.out.name
    torch.onnx.export(
        model,
        (dummy,),
        str(raw),
        input_names=["obs"],
        output_names=["policy", "wdl", "value"],
        # Batch stays dynamic even though the page always evaluates one leaf at
        # a time: the parity harness pushes a whole suite through in one call.
        dynamic_axes={
            "obs": {0: "batch"},
            "policy": {0: "batch"},
            "wdl": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    import onnx

    m = onnx.load(str(raw))  # pulls any sidecar back into memory

    if args.fp16:
        from onnxconverter_common import float16


        # keep_io_types: the page hands in float32 observations and reads
        # float32 logits. Casting at the boundary costs 7616 + 4672 elements
        # and saves the JS from having to know the model's internal precision.
        m = float16.convert_float_to_float16(m, keep_io_types=True)
        # The converter rewrites tensor types but leaves the graph's cached
        # `value_info` claiming float for nodes it just made float16, and ONNX
        # Runtime refuses to load the contradiction. These are inferable hints,
        # not data: drop them and let shape inference rebuild them.
        del m.graph.value_info[:]
        m = onnx.shape_inference.infer_shapes(m, strict_mode=False)

    onnx.save(m, str(args.out), save_as_external_data=False)
    for f in staging.iterdir():
        f.unlink()
    staging.rmdir()

    params = sum(p.numel() for p in net.parameters())
    _, _, meta = load_checkpoint(args.checkpoint, "cpu")
    manifest = {
        "checkpoint": provenance(args.checkpoint),
        "gen": meta.get("gen"),
        "run": args.checkpoint.parent.name,
        "sha256": sha256(args.out),
        "bytes": args.out.stat().st_size,
        "precision": "fp16" if args.fp16 else "fp32",
        "opset": args.opset,
        "arch": arch,
        "params": params,
        "planes": PLANES,
        "board": list(BOARD),
        "policy_len": POLICY,
    }
    (args.out.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    mb = args.out.stat().st_size / 1e6
    print(f"{args.out}  {mb:.1f} MB  {params/1e6:.2f}M params  {manifest['precision']}")
    print(f"  arch {arch}")


if __name__ == "__main__":
    main()
