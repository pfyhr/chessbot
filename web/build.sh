#!/usr/bin/env bash
# Assemble the GitHub Pages site into web/dist.
#
# Everything the page serves is produced here, so the deployed site and the
# thing the parity test checks are the same bytes. Run `tools/parity_check.mjs`
# against the result before publishing -- the build will happily produce a site
# with a transposed encoding or a half-converted model, and both look fine
# until you notice the network is playing like it was never trained.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
WEB=$REPO/web
DIST=${DIST:-$WEB/dist}
CKPT=${CKPT:-$REPO/runs/long-oct/gen2600.pt}
PY=${PY:-$REPO/.venv/bin/python}
ORT=$WEB/node_modules/onnxruntime-web/dist

say () { printf '\033[1m==> %s\033[0m\n' "$*"; }

say "search core -> wasm32"
# Niced: this machine shares its cores with other people's jobs, and a build is
# never the most urgent thing on it.
( cd "$REPO" && nice -n 15 cargo build -p az-wasm --target wasm32-unknown-unknown --release )

# The model half needs PyTorch and the 131 MB training checkpoint, neither of
# which exists on a CI runner -- the checkpoint is far too big for the repo. So
# CI sets SKIP_MODEL=1 and fetches the already-exported model instead, and the
# reference vectors it is checked against are committed (they are 170 KB).
# Exporting and checking stay the same job locally, which is where it matters:
# the model and the numbers that gate it are produced by one command.
if [ "${SKIP_MODEL:-0}" = "1" ]; then
  say "network: using the pre-exported model in $DIST/model"
  test -f "$DIST/model/gen2600.onnx" || { echo "missing $DIST/model/gen2600.onnx"; exit 1; }
  test -f "$DIST/model/selfcheck.json" || { echo "missing $DIST/model/selfcheck.json"; exit 1; }
else
  say "network -> ONNX (fp16)"
  # fp16 halves a 44 MB download. Justified by measurement, not by hope: see
  # bench/results/ and the gate in tools/parity_check.mjs.
  PYTHONPATH=$REPO/python nice -n 15 "$PY" "$WEB/tools/export_onnx.py" "$CKPT" \
    --out "$DIST/model/gen2600.onnx" --fp16

  say "reference vectors"
  PYTHONPATH=$REPO/python nice -n 15 "$PY" "$WEB/tools/parity_ref.py" "$CKPT" \
    --onnx "$DIST/model/gen2600.onnx" \
    --out "$WEB/parity/reference.json" \
    --selfcheck "$DIST/model/selfcheck.json"
fi

say "static files"
mkdir -p "$DIST/src" "$DIST/vendor/ort"
cp "$WEB/index.html" "$DIST/"
cp "$WEB"/src/*.js "$WEB"/src/*.css "$DIST/src/"
# The piece font. Six glyphs, 1.7 KB -- see the @font-face note in
# style.css for why the board cannot rely on system fallback.
cp "$WEB"/src/chess-pieces.woff2 "$DIST/src/"
cp "$REPO/target/wasm32-unknown-unknown/release/az_wasm.wasm" "$DIST/"

say "ONNX Runtime"
# Three files, and all three are needed. `ort.bundle.min.mjs` dynamically
# imports the `.jsep.mjs` glue, which in turn fetches the `.jsep.wasm` binary --
# the bundle does NOT inline the glue, despite the name. Shipping only the
# bundle and the binary gets you "no available backend found" at runtime, with
# everything else on the page working perfectly.
#
# The `jsep` binary carries the WebGPU backend and the CPU kernels together, so
# one file serves both paths and nobody downloads a backend they cannot use.
cp "$ORT/ort.bundle.min.mjs" "$DIST/vendor/ort/"
cp "$ORT/ort-wasm-simd-threaded.jsep.mjs" "$DIST/vendor/ort/"
cp "$ORT/ort-wasm-simd-threaded.jsep.wasm" "$DIST/vendor/ort/"

# Pages serves no custom headers, so there is no COOP/COEP and therefore no
# SharedArrayBuffer and no wasm threads. Saying so in a file beats having the
# runtime discover it and warn on every load.
touch "$DIST/.nojekyll"

say "done"
du -sh "$DIST"
find "$DIST" -type f | sort | while read -r f; do
  printf '  %7s  %s\n' "$(du -h "$f" | cut -f1)" "${f#"$DIST"/}"
done
