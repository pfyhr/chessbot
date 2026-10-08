#!/bin/bash
# Provision a rented GPU box and start the width-against-budget sweep.
#
#   ./bench/vast_deploy.sh 1.2.3.4 8640 check   # inspect only, install nothing
#   ./bench/vast_deploy.sh 1.2.3.4 8640         # ship, build, calibrate, launch
#
# Ships source and one checkpoint rather than cloning: the work is on the laptop
# and not pushed. Two things about Vast containers that this handles and a naive
# script does not:
#
#   * nproc reports the HOST's cores, not our slice. A box advertising 12.8
#     cores reports 64. The cgroup quota is the real number, and it decides how
#     many engine processes we can feed before the drivers starve the GPU.
#   * the PyTorch template keeps torch in its own venv (/venv/main), not in the
#     system python, so the base interpreter has to be discovered.
set -eu
HOST=${1:?usage: vast_deploy.sh <host> <port> [check]}
PORT=${2:?usage: vast_deploy.sh <host> <port> [check]}
MODE=${3:-run}
R=$(cd "$(dirname "$0")/.." && pwd)
SSHOPTS="-o StrictHostKeyChecking=accept-new -o LogLevel=ERROR"
SSH="ssh -p $PORT $SSHOPTS root@$HOST"

say () { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "what we rented"
$SSH 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
      echo "ram: $(free -g | awk "/Mem:/{print \$2}")G   disk free: $(df -h / | awk "NR==2{print \$4}")"'

CORES=$($SSH 'if [ -f /sys/fs/cgroup/cpu.max ]; then
                read q p < /sys/fs/cgroup/cpu.max
                if [ "$q" = "max" ]; then nproc; else echo $((q/p)); fi
              elif [ -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
                q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
                p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
                if [ "$q" -le 0 ]; then nproc; else echo $((q/p)); fi
              else nproc; fi' | tr -d '\r')

BASEPY=$($SSH 'for c in /venv/main/bin/python /opt/conda/bin/python /usr/bin/python3; do
                 if [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null; then echo "$c"; exit 0; fi
               done; echo NONE' | tr -d '\r')
[ "$BASEPY" = "NONE" ] && { echo "ERROR: no python with torch. Use the PyTorch (Vast) template."; exit 1; }
$SSH "$BASEPY -c 'import sys,torch; print(\"base python %d.%d\" % sys.version_info[:2], \"| torch\", torch.__version__, \"| cuda\", torch.cuda.is_available())'"

# fastchess runs CONC games at once, but chess is turn-based: only one engine
# per game is searching at any moment. So the CPU-bound work is CONC threads,
# not 2*CONC -- the other half sit idle waiting for their turn. Match
# concurrency to cores, not to half of them. (Measured: at CONC=8 on 12 cores
# the box ran 8 active searches and sat a third idle.)
CONC=${CONC:-$CORES}
[ "$CONC" -lt 4 ] && CONC=4
[ "$CONC" -gt 16 ] && CONC=16
echo "cores: $CORES (cgroup, not nproc)  ->  concurrency $CONC"
echo "  $CONC active searches, $((CONC*2)) processes ($((CONC)) idle awaiting their turn)"
[ "$MODE" = "check" ] && { echo; echo "check only; nothing installed."; exit 0; }

say "shipping source and gen695"
$SSH 'mkdir -p ~/chessbot/runs/champ'
rsync -az -e "ssh -p $PORT $SSHOPTS" \
      "$R/crates" "$R/python" "$R/bench" "$R/Cargo.toml" "$R/pyproject.toml" \
      root@"$HOST":chessbot/
"$R/.venv/bin/python" - "$R" <<'SLIM'
import sys, pathlib, torch
src = pathlib.Path(sys.argv[1]) / "runs/w-long/gen695.pt"
dst = pathlib.Path("/tmp/gen695-slim.pt")
if not dst.exists():
    ck = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({"net": ck["net"], "meta": ck.get("meta", {})}, dst)
print(f"checkpoint {dst.stat().st_size/1e6:.0f}MB")
SLIM
rsync -az -e "ssh -p $PORT $SSHOPTS" /tmp/gen695-slim.pt root@"$HOST":chessbot/runs/champ/gen695.pt

say "building (rust, bridge, fastchess)"
cat > /tmp/remote_setup.sh <<'REMOTE'
set -e
cd ~/chessbot
command -v cargo >/dev/null 2>&1 || curl -sSf https://sh.rustup.rs | sh -s -- -y >/dev/null
. "$HOME/.cargo/env"
# --system-site-packages so the image's torch is reused, not re-downloaded
[ -d .venv ] || "$BASEPY" -m venv --system-site-packages .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q maturin "chess>=1.11"
.venv/bin/pip install -q -e . --no-build-isolation 2>&1 | tail -1 || true
.venv/bin/maturin develop --release 2>&1 | tail -1
if [ ! -x ~/fastchess/fastchess ]; then
  git clone -q https://github.com/Disservin/fastchess.git ~/fastchess
  make -C ~/fastchess -j"$(nproc)" >/dev/null 2>&1
fi
echo "built: fastchess $(du -h ~/fastchess/fastchess | cut -f1), bridge $(.venv/bin/python -c 'import chessbot_core;print(chessbot_core.POLICY_LEN)') policy slots"
REMOTE
scp -q -P "$PORT" $SSHOPTS /tmp/remote_setup.sh root@"$HOST":/tmp/
$SSH "BASEPY='$BASEPY' bash /tmp/remote_setup.sh"

say "smoke test"
$SSH 'cd ~/chessbot && printf "uci\nquit\n" | .venv/bin/chess-uci --run ~/chessbot/runs/champ 2>&1 | grep -E "^id name|Device"'

say "calibration: $((CONC*2)) games at 512 nodes (two full waves)"
CAL=$($SSH "cd ~/chessbot && export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && t0=\$SECONDS
  ~/fastchess/fastchess \
    -engine cmd=\$HOME/chessbot/.venv/bin/chess-uci args=\"--run \$HOME/chessbot/runs/champ\" \
      name=a option.Generation=695 option.RootActions=8 nodes=512 \
    -engine cmd=\$HOME/chessbot/.venv/bin/chess-uci args=\"--run \$HOME/chessbot/runs/champ\" \
      name=b option.Generation=695 option.RootActions=32 nodes=512 \
    -rounds $CONC -games 2 -concurrency $CONC -srand 1 \
    -openings file=\$HOME/chessbot/bench/openings-4ply.epd format=epd order=random \
    -log file=/dev/null >/tmp/cal.log 2>&1
  echo \$((SECONDS-t0))" | tr -d '\r' | tail -1)
# Project from THIS box's measured throughput rather than a ratio to a stale
# reference. Two full waves of CONC games, so no idle slots skew it -- sizing it
# at 16 games against concurrency 12 ran 12 then 4 and read 1.5x pessimistic.
#
# The sweep is 2,800 game-equivalents at 512-node cost:
#   512:  2 duels x 200 games x 1  =  400
#   1024: 3 duels x 200 games x 2  = 1200
#   2048: 3 duels x 100 games x 4  = 1200
awk -v s="$CAL" -v c="$CONC" 'BEGIN{
  wave = s/2;                 # seconds for one full wave of c games
  per  = wave/c;              # seconds per game-equivalent at full occupancy
  t    = 2800*per/3600;
  printf "  %ss for 2 waves of %d games -> %.1fs per game at full occupancy\n", s, c, per;
  printf "  projected sweep: %.0fh\n", t;
  if (t>80)      print "  SLOW -- worth taking another offer.";
  else if (t>45) print "  usable; consider dropping the 2048 rung.";
  else           print "  nominal. Go.";
}'

say "starting the sweep in tmux"
$SSH "cd ~/chessbot && tmux kill-session -t sweep 2>/dev/null || true
      tmux new-session -d -s sweep 'CONC=$CONC bash ~/chessbot/bench/width_budget_sweep.sh'
      sleep 3 && tmux ls"
echo
echo "watch:  ssh -p $PORT root@$HOST 'tail -f ~/chessbot/runs/width-budget/report.log'"
