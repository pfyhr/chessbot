#!/bin/bash
# Publish a checkpoint as a GitHub release so training can continue elsewhere.
#
#   ./bench/publish_champion.sh w-long 695
#
# Three things are needed to continue a run on a different machine, and only the
# first is portable today:
#
#   1. the code            -- git, already fine
#   2. the FULL checkpoint -- net AND optimizer state. --init without the
#      optimizer restarts AdamW with zeroed moments, which perturbs an already
#      converged network rather than continuing it.
#   3. the training arguments -- otherwise a "continuation" silently trains a
#      different experiment. `considered` was never logged before 2026-10-08, so
#      for older checkpoints it is recorded here as unknown rather than guessed.
#
# This ships all three. The manifest is the part that makes it reproducible.
set -eu
RUN=${1:?usage: publish_champion.sh <run-dir-name> <generation>}
GEN=${2:?usage: publish_champion.sh <run-dir-name> <generation>}
R=$(cd "$(dirname "$0")/.." && pwd)
CK="$R/runs/$RUN/gen$(printf '%03d' "$GEN").pt"
TAG="champion-${RUN}-g${GEN}"
WORK=$(mktemp -d)
[ -f "$CK" ] || { echo "no such checkpoint: $CK"; exit 1; }

"$R/.venv/bin/python" - "$R" "$RUN" "$GEN" "$WORK" <<'PY'
import json, sys, pathlib, torch, subprocess
R, run, gen, work = sys.argv[1], sys.argv[2], int(sys.argv[3]), pathlib.Path(sys.argv[4])
d = pathlib.Path(R) / "runs" / run
h = json.load(open(d / "history.json"))[: gen + 1]
ck = torch.load(d / f"gen{gen:03d}.pt", map_location="cpu", weights_only=True)
log = (d / "train.log").read_text().splitlines()[:6] if (d / "train.log").exists() else []

# `considered` only entered the log on 2026-10-08; never invent it.
considered = next((l.split("considered=")[1].split()[0] for l in log if "considered=" in l), None)

manifest = {
    "tag": f"champion-{run}-g{gen}",
    "run": run, "generation": gen,
    "commit": subprocess.run(["git", "-C", R, "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip(),
    "architecture": {
        "blocks": 6, "channels": 96, "policy_head": "fc", "trunk": "res",
        "policy_bottleneck": 32,
        "parameters": sum(v.numel() for v in ck["net"].values()),
    },
    "training": {
        "generations_this_run": len(h),
        "games_this_run": sum(r["games"] for r in h),
        "evaluations_this_run": sum(r["evaluations"] for r in h),
        "hours_this_run": round(h[-1]["elapsed_total"] / 3600, 1),
        "sims": next((l.split("sims=")[1].split()[0] for l in log if "sims=" in l), None),
        "considered": considered,
        "considered_note": None if considered else
            "not recorded: the header only printed it from 2026-10-08. Use 32, "
            "the documented self-play setting, unless reproducing this exactly.",
        "lineage": log[0] if log and "resuming" in log[0] else "trained from scratch",
    },
    "losses": {"policy": round(h[-1]["policy_loss"], 4),
               "value": round(h[-1]["value_loss"], 4)},
    "checkpoint": {"keys": list(ck.keys()),
                   "has_optimizer_state": "opt" in ck and ck["opt"] is not None},
}
(work / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest["training"], indent=2))
PY

cp "$CK" "$WORK/gen$(printf '%03d' "$GEN").pt"
NOTES="$WORK/notes.md"
cat > "$NOTES" <<EOF
An AlphaZero-style chess network trained from zero, no human games, no opening
book, no handcrafted evaluation. 6x96 residual trunk, 10.99M parameters.

**This checkpoint carries optimizer state**, so training continues rather than
restarts. \`--init\` without it rebuilds AdamW with zeroed moments, which
perturbs a converged network instead of continuing it.

## Continue training from it

\`\`\`bash
gh release download $TAG --repo pfyhr/chessbot
pip install maturin && maturin develop --release
# resume into a FRESH directory -- training starts history at [] and generation
# numbering at 0, so pointing --out at the source run destroys it
chess-train --init gen$(printf '%03d' "$GEN").pt --out runs/continued \\
            --sims 32 --considered 32 --keep-every 25 --max-hours 24
\`\`\`

## Play against it

\`\`\`bash
mkdir -p runs/champ && mv gen$(printf '%03d' "$GEN").pt runs/champ/
chess-uci --run runs/champ          # UCI engine, any GUI
\`\`\`

\`manifest.json\` records the architecture, the training arguments, the commit
it was produced at, and its lineage. Where a setting was not logged at the time
it says so rather than guessing.

MIT licensed, as the rest of the repository.
EOF

echo
echo "publishing $TAG ..."
gh release create "$TAG" "$WORK/gen$(printf '%03d' "$GEN").pt" "$WORK/manifest.json" \
   --repo pfyhr/chessbot --title "Champion $RUN gen$GEN" --notes-file "$NOTES" \
  || gh release upload "$TAG" "$WORK/gen$(printf '%03d' "$GEN").pt" "$WORK/manifest.json" \
     --repo pfyhr/chessbot --clobber
rm -rf "$WORK"
echo "done: https://github.com/pfyhr/chessbot/releases/tag/$TAG"
