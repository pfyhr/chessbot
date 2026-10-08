#!/bin/bash
# Provision a rented GPU box and start the width-against-budget sweep.
#
#   ./bench/vast_deploy.sh ssh4.vast.ai 12345        # host and port from Vast
#   ./bench/vast_deploy.sh ssh4.vast.ai 12345 check  # just verify, start nothing
#
# Ships source and one checkpoint rather than cloning: the work is on this
# laptop and not pushed. Assumes a PyTorch image, so torch comes from the system
# python and the venv inherits it.
set -eu
HOST=${1:?usage: vast_deploy.sh <host> <port> [check]}
PORT=${2:?usage: vast_deploy.sh <host> <port> [check]}
MODE=${3:-run}
R=$(cd "$(dirname "$0")/.." && pwd)
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new root@$HOST"
RS="-e ssh -p $PORT -o StrictHostKeyChecking=accept-new"

say () { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "what we rented"
$SSH 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
      echo "cores: $(nproc)   ram: $(free -g | awk "/Mem:/{print \$2}")G"
      echo "disk free: $(df -h / | awk "NR==2{print \$4}")"
      python3 -c "import torch;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available())" 2>/dev/null || echo "NO TORCH"'

# The driver is single-threaded per engine process and we run 2*CONC of them,
# so cores are the thing that silently halves throughput on a cheap offer.
CORES=$($SSH 'nproc' | tr -d '\r')
if [ "$CORES" -lt 16 ]; then
  echo "WARNING: only $CORES cores. Each of the 2 x concurrency engine processes"
  echo "         needs one. Consider CONC=$((CORES/2 - 1)) or a better offer."
fi
[ "$MODE" = "check" ] && { echo "check only; nothing started"; exit 0; }

say "shipping source and gen695"
$SSH 'mkdir -p ~/chessbot/runs/champ'
rsync -az --rsync-path="mkdir -p ~/chessbot && rsync" $RS \
      "$R/crates" "$R/python" "$R/bench" "$R/Cargo.toml" "$R/pyproject.toml" \
      root@"$HOST":~/chessbot/
python3 - "$R" <<'PY'
import sys, torch, pathlib
src = pathlib.Path(sys.argv[1]) / "runs/w-long/gen695.pt"
dst = pathlib.Path("/tmp/gen695-slim.pt")
if not dst.exists():
    ck = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({"net": ck["net"], "meta": ck.get("meta", {})}, dst)
print("slim checkpoint", dst, f"{dst.stat().st_size/1e6:.0f}MB")
PY
rsync -az --progress $RS /tmp/gen695-slim.pt root@"$HOST":~/chessbot/runs/champ/gen695.pt

say "building"
$SSH 'set -e
  cd ~/chessbot
  command -v cargo >/dev/null || { curl -sSf https://sh.rustup.rs | sh -s -- -y >/dev/null; }
  . "$HOME/.cargo/env"
  # --system-site-packages so the image torch is reused rather than re-downloaded
  [ -d .venv ] || python3 -m venv --system-site-packages .venv
  .venv/bin/pip install -q maturin "chess>=1.11" numpy
  .venv/bin/pip install -q -e . --no-build-isolation 2>/dev/null || true
  .venv/bin/maturin develop --release 2>&1 | tail -2
  [ -x ~/fastchess/fastchess ] || {
    git clone -q https://github.com/Disservin/fastchess.git ~/fastchess
    make -C ~/fastchess -j"$(nproc)" >/dev/null 2>&1; }
  echo "fastchess: $(ls -la ~/fastchess/fastchess | wc -l) built"'

say "smoke test: does the engine come up on the GPU?"
$SSH 'cd ~/chessbot && printf "uci\nquit\n" | .venv/bin/chess-uci --run ~/chessbot/runs/champ 2>&1 | grep -E "^id name|Device"'

say "starting the sweep in tmux"
$SSH 'cd ~/chessbot && tmux kill-session -t sweep 2>/dev/null || true
      tmux new-session -d -s sweep "bash ~/chessbot/bench/width_budget_sweep.sh"
      sleep 3 && tmux ls'
echo
echo "watch it with:  ssh -p $PORT root@$HOST 'tail -f ~/chessbot/runs/width-budget/report.log'"
