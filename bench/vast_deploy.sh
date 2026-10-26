#!/bin/bash
# Provision a rented GPU box and start the width-against-budget sweep.
#
#   ./bench/vast_deploy.sh 1.2.3.4 8640 check    # inspect only, install nothing
#   ./bench/vast_deploy.sh 1.2.3.4 8640 measure  # build, then run docs/rented-training-plan.md section 1
#   ./bench/vast_deploy.sh 1.2.3.4 8640 probe    # build, then measure both A/B arms (needs 2 cards)
#   ./bench/vast_deploy.sh 1.2.3.4 8640 ab       # build and launch the sims A/B
#   ./bench/vast_deploy.sh 1.2.3.4 8640          # ship, build, calibrate, launch sweep
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

# EGRESS TEST, before committing hours to this box. On 2026-10-21 a host
# advertising 857 Mbps up delivered 0.2 Mbps to Sweden -- 49h for one run's
# checkpoints. Uploads from that same box to Cloudflare ran at 31-39 Mbps and
# downloads INTO it at 54, so neither the listing, DLPerf, nor a download test
# would have caught it. The only number that matters is bytes per second from
# the box to the machine that will hold the results.
#
# This runs inside `check`, before anything is installed. It used to run after
# the build, which is the wrong order: section 1 of the plan says egress is
# measured first precisely so that a bad box is abandoned before it has cost
# twenty minutes of rent.
egress_test () {
  say "egress: can we actually get results off this box?"
  $SSH 'dd if=/dev/urandom of=/tmp/egress.bin bs=1M count=40 2>/dev/null'
  local t0=$(date +%s)
  $SSH -o Compression=no 'cat /tmp/egress.bin' > /tmp/egress_local.bin 2>/dev/null
  local el=$(( $(date +%s) - t0 ))
  local sz=$(wc -c < /tmp/egress_local.bin | tr -d ' ')
  $SSH 'rm -f /tmp/egress.bin'; rm -f /tmp/egress_local.bin
  awk -v sz="$sz" -v el="$el" 'BEGIN{
    if (el < 1) el = 1;
    mbps = sz*8/el/1000000;
    printf "  %.0f MB in %ds = %.1f Mbps to here\n", sz/1048576, el, mbps;
    gb = 4.4;
    printf "  a run\x27s %.1f GB of checkpoints would take %.1f h at this rate\n", gb, gb*1024*8/mbps/3600;
    if (mbps < 5)       print "  KILL CRITERION HIT (<5 Mbps) -- the results cannot come home. Take another offer.";
    else if (mbps < 10) print "  below the 10 Mbps bar in section 1. Usable only if checkpoints are pulled inline.";
    else if (mbps < 50) print "  passes, but pull checkpoints inline rather than at the end.";
    else                print "  fine.";
  }'
}


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
if [ "$MODE" = "check" ]; then
  egress_test
  echo
  echo "check only; nothing installed."
  exit 0
fi

# The training A/B is GPU-bound, so unlike the match sweep it wants a fast card
# and few cores. `probe` measures the real per-generation cost of both arms
# sharing one GPU, which is the only number that decides whether a box is worth
# 24h. Costs about ten minutes and five cents.

say "shipping source"
$SSH 'mkdir -p ~/chessbot/runs/champ'
rsync -az -e "ssh -p $PORT $SSHOPTS" \
      "$R/crates" "$R/python" "$R/bench" "$R/Cargo.toml" "$R/pyproject.toml" \
      root@"$HOST":chessbot/

# The box fetches the champion itself from the GitHub release rather than
# waiting on this laptop to be awake and reachable. The release carries the FULL
# checkpoint -- optimizer state included -- so a continuation really continues.
CHAMPION=${CHAMPION:-champion-w-long-g695}
say "fetching $CHAMPION from the release"
$SSH "cd ~/chessbot/runs/champ && \
  curl -sSL -o gen695.pt \
    https://github.com/pfyhr/chessbot/releases/download/$CHAMPION/gen695.pt && \
  curl -sSL -o manifest.json \
    https://github.com/pfyhr/chessbot/releases/download/$CHAMPION/manifest.json && \
  ls -la gen695.pt | awk '{print \"  \" \$5 \" bytes\"}' && \
  python3 -c \"import json;m=json.load(open('manifest.json'));print('  gen',m['generation'],'| optimizer state:',m['checkpoint']['has_optimizer_state'])\""

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

egress_test

say "smoke test"
$SSH 'cd ~/chessbot && printf "uci\nquit\n" | .venv/bin/chess-uci --run ~/chessbot/runs/champ 2>&1 | grep -E "^id name|Device"'

if [ "$MODE" = "measure" ]; then
  # Section 1 of docs/rented-training-plan.md, scripted end to end so the box is
  # never idle while someone decides what to run next. ONE card; `probe` is the
  # two-card A/B and is a different question.
  #
  # Order matters and is the plan's: egress (already run above), real core
  # count (already printed above), driver ceiling, then the generation profile
  # across precisions, then the batch sweep. Each phase prints the threshold it
  # is read against, so the decision is made against what was written down
  # rather than against how much has already been spent.
  say "section 1.3/1.4: driver ceiling, network excluded"
  $SSH "cd ~/chessbot && . \$HOME/.cargo/env && \
    cargo build --release -p az-core --bin drivercost 2>&1 | tail -1 && \
    for t in 1 2 4 8; do
      ./target/release/drivercost --sims 32 --concurrency 512 --games 2048 --threads \$t \
        | grep -E 'evals/sec \(|games/sec' | sed \"s/^/  threads \$t  /\"
    done"
  echo "  Jupiter, one thread: 292,646 evals/s."
  echo "  KILL CRITERION: single-thread < 120,000 evals/s and the core is too slow"
  echo "  for the driver whatever card is bolted to it."

  say "section 1.7: all four precision regimes"
  # The whole hardware ranking turns on this. TF32 is worth nothing on a GeForce
  # card and 7.4x on an H100, and torch turns TF32 convolutions on by DEFAULT --
  # so "default" is what every previous CUDA measurement actually ran, and
  # "strict" is the only honest fp32. The profiler prints the flags torch reports
  # back, not the ones it was asked for.
  for prec in strict default tf32 bf16; do
    echo "  --- $prec ---"
    $SSH "cd ~/chessbot && .venv/bin/python bench/generation_profile.py \
      --device cuda --games 256 --concurrency 256 --sims 32 --considered 32 \
      --steps 250 --precision $prec --driver-threads ${DT:-4}" \
      > /tmp/prof-$prec.txt 2>&1
    grep -E 'precision:|gpu:|generation total|GPU-bound|driver seconds|inference throughput' \
      /tmp/prof-$prec.txt | sed 's/^/    /'
  done

  say "section 1.6: torch.compile on CUDA, on and off"
  for flag in "" "--compile"; do
    $SSH "cd ~/chessbot && .venv/bin/python bench/generation_profile.py \
      --device cuda --games 256 --concurrency 256 --sims 32 --considered 32 \
      --steps 250 --precision default --driver-threads ${DT:-4} $flag" \
      2>&1 | grep -E 'compile=|generation total' | sed 's/^/    /'
  done
  echo "  worth 1.36x on Jupiter; CUDA compile times are longer."

  say "section 1.8: batch sweep -- compute-bound or bandwidth-bound?"
  # The measurement that separates the two regimes the plan cannot choose
  # between. Still climbing at the top  -> launch-bound, raise concurrency.
  # Same plateau in both precisions     -> bandwidth-bound, and the big cards'
  #                                        tensor peaks are unreachable.
  # bf16 lifts the plateau              -> compute-bound, tensor cores pay.
  for prec in strict bf16; do
    echo "  --- $prec ---"
    $SSH "cd ~/chessbot && .venv/bin/python bench/generation_profile.py \
      --device cuda --precision $prec --sweep 256,512,1024,2048,4096" \
      2>&1 | tail -12 | sed 's/^/    /'
  done

  say "read it against what was declared"
  cat <<'THRESH'
  Jupiter is 85.2 s/generation, 42.3 generations/hour.

  THIS BASELINE WAS CORRECTED ON 2026-10-26 and the thresholds moved with it.
  It used to read 97.0 s / 37.1 per hour, taken from runs/long-oct -- but every
  generation there carries evaluated:true, so 97.0 INCLUDES the ~12 s probe
  suite. The numbers measured below come from generation_profile.py, which
  reports "generation total (no eval)". Reading a no-probe number against a
  with-probe baseline flattered every candidate box by ~14%, enough to flip a
  verdict: 45 s/generation is 2.16x against 97.0 and survives, 1.89x against
  85.2 and is killed. 85.2 s is the like-for-like figure, measured the same day
  by bench/driver_threads.sh.

  WORTH RENTING if, in the best precision regime that passes a quality check:
    generations/hour  >= 3.0x Jupiter  (>= 127/h, i.e. <= 28.4 s/generation)
    GPU-bound fraction >= 65%
    egress            >= 10 Mbps
  KILLS IT if:
    generations/hour  <  2.0x Jupiter  (<  85/h, i.e. >= 42.6 s/generation)
    GPU-bound fraction <  50%
    egress            <  5 Mbps
    drivercost 1-thread < 126,000 evals/s   (0.4x Jupiter's measured 314,542)

  bf16 is a SPEED number only until a loss curve and a match say otherwise
  (section 1.9). Do not read a bf16 result against the thresholds before then.
THRESH
  echo
  echo "measurement only; nothing launched. Destroy this instance when done."
  exit 0
fi

if [ "$MODE" = "probe" ]; then
  say "does this card actually have kernels, or will it JIT?"
  $SSH "$BASEPY -c '
import torch
arch = torch.cuda.get_arch_list()
n = torch.cuda.device_count()
print(\"built for:\", \" \".join(arch))
names = set()
for i in range(n):
    cap = torch.cuda.get_device_capability(i)
    nm = torch.cuda.get_device_name(i)
    names.add(nm)
    # A cubin is forward-compatible within a major compute capability: an
    # sm_86 binary loads natively on an sm_89 device. Checking for an exact
    # sm_XY match calls a perfectly good card a JIT risk -- torch 2.14 ships
    # sm_86 but not sm_89, and a 4090 runs it natively.
    usable = [x for x in arch
              if x.startswith(\"sm_\")
              and int(x[3:-1] or 0) == cap[0] and int(x[-1]) <= cap[1]]
    best = max(usable, key=lambda x: int(x[-1])) if usable else None
    print(\"  gpu%d\" % i, nm, \"| capability %d.%d\" % cap,
          \"| native kernels:\", (\"YES via \" + best) if best else \"NO -- will JIT from PTX\")
print(\"devices:\", n, \"| identical:\", \"YES\" if len(names) == 1 else \"NO -- a hardware difference would confound the arms\")
if n < 2: print(\"ERROR: the A/B needs two cards, one per arm\")
'"

  say "both A/B arms, ONE PER CARD, one generation each"
  # Not sharing a card: that is the design that deadlocked on 2026-10-09. This
  # also measures the real per-generation cost under the CPU contention the
  # actual run will have, which is the number that decides the box.
  $SSH "cd ~/chessbot && \
    ( CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench/generation_profile.py --games 256 --steps 250 --sims 32 >/tmp/p32.txt 2>&1 & \
      CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/generation_profile.py --games 256 --steps 250 --sims 64 >/tmp/p64.txt 2>&1 & \
      wait )
    for f in /tmp/p32.txt /tmp/p64.txt; do
      echo \"--- \$f ---\"; grep -E 'generation total|GPU-bound|network forward|Rust driver' \$f
    done" | tee /tmp/probe_out.txt

  G32=$(grep -A4 "p32" /tmp/probe_out.txt | grep "generation total" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
  G64=$(grep -A4 "p64" /tmp/probe_out.txt | grep "generation total" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
  awk -v a="$G32" -v b="$G64" 'BEGIN{
    if (a=="" || b=="") { print "\n  could not parse; read the output above"; exit }
    printf "\n  per generation: sims32 %.1fs  sims64 %.1fs  (ratio %.2fx)\n", a, b, b/a;
    printf "  in 24h: %.0f and %.0f generations\n", 86400/a, 86400/b;
    printf "  the capacity A/B that settled its question had 575 and 386\n";
    if (86400/b < 300) print "  THIN -- the 64-sims arm lands near the short-run regime. Prefer a faster card.";
    else               print "  enough. Good box for the A/B.";
  }'
  echo
  echo "probe only; nothing launched. Destroy this instance when done."
  exit 0
fi

if [ "$MODE" = "ab" ]; then
  # The match calibration below sizes a fastchess sweep and tells us nothing
  # about a training run. Skip straight to the launch.
  :
else
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
fi

if [ "$MODE" = "ab" ]; then
  say "starting the sims A/B in tmux (${HOURS:-12}h, one arm per card)"
  $SSH "cd ~/chessbot && tmux kill-session -t ab 2>/dev/null || true
        tmux new-session -d -s ab 'HOURS=${HOURS:-12} GPU32=0 GPU64=1 bash ~/chessbot/bench/sims_ab.sh'
        sleep 5 && tmux ls && tail -5 ~/chessbot/runs/sims-ab.log 2>/dev/null"
  echo
  echo "watch:   ssh -p $PORT root@$HOST 'tail -f ~/chessbot/runs/ab-sims32.log'"
  echo "progress: ssh -p $PORT root@$HOST \"stat -c %y ~/chessbot/runs/ab-sims*.log\""
  echo
  echo "Check PROGRESS, not liveness -- on 2026-10-09 a wedged arm passed every"
  echo "liveness check for 9.7h. The script's watchdog aborts both arms after"
  echo "20min of no advance, but verify the log mtimes are moving yourself too."
  exit 0
fi

say "starting the sweep in tmux"
$SSH "cd ~/chessbot && tmux kill-session -t sweep 2>/dev/null || true
      tmux new-session -d -s sweep 'CONC=$CONC bash ~/chessbot/bench/width_budget_sweep.sh'
      sleep 3 && tmux ls"
echo
echo "watch:  ssh -p $PORT root@$HOST 'tail -f ~/chessbot/runs/width-budget/report.log'"
