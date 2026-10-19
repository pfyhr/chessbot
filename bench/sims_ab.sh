#!/bin/bash
# Does a bigger self-play search budget train a better network? -- rented GPU.
#
# Everything measured so far about search budget concerns MATCH play on an
# already-trained network: +539 Elo across the ladder, the root-width optimum,
# all of it. Self-play search does a second job those cannot see -- it produces
# the improved-policy training target -- and the two demonstrably come apart
# (widening the root won +102 Elo of trained strength while making the tree
# shallower). This is the instrument that can see the second job.
#
# Design, following the capacity A/B that settled its own question:
#   * both arms run CONCURRENTLY on the same BOX, one card each, so contention
#     is identical. An arm that runs alone cannot be compared against one that
#     shared -- that is what made cap-xl uninterpretable. (Revised 2026-10-19:
#     the first attempt shared ONE card and deadlocked; see the GPU check below.)
#   * wall-clock matched, not generation matched. The question is what a budget
#     buys per hour, and the 64-sims arm pays for its search in games forgone.
#   * considered=32 on both, stated explicitly: it is worth ~+102 Elo and was
#     never recorded in any past run's log.
#   * keep-every 50: a checkpoint is ~126 MB, every generation would be 113 GB,
#     and even every 25th is 11.6 GB against the 16 GB this box has free.
#
# Probed on an RTX 5080: 60.5s and 98.8s per generation, so 24h gives roughly
# 1,428 and 874 generations against the 575/386 the capacity A/B worked from.
#
# What this rerun is for (2026-10-19). The salvaged first attempt answered the
# question out to 5h: sims32 wins per hour by +325 at 1.5h, +147 at 2.8h, and is
# level at 5.0h. Those three points are log-linear at -704 Elo per decade, which
# would put sims64 ~300 Elo ahead at 12h -- the regime we actually train in.
# That is a 2.4x extrapolation from three points against two parameters, so it
# is a prediction to test, not a result. 12h is chosen to reach well past the
# predicted 4.5h crossover. See bench/results/2026-10-19-sims-crossover.md.
#
# NOTE the new run does not extend the old curve and must not be spliced onto
# it: the old hours were measured with both arms sharing a card, these are with
# a card each, so the cost basis differs.
set -u
R=$HOME/chessbot
OUT=$R/runs
HOURS=${HOURS:-12}
mkdir -p "$OUT"
cd "$R" || exit 1

# ONE ARM PER CARD. The 9 Oct attempt put both arms on one GPU for contention
# symmetry and lost 9.7h to a driver deadlock: two CUDA contexts competing for
# one driver is what appears to have wedged it. A two-GPU box keeps contention
# symmetric by construction -- same host, same CPU, same RAM, one card each --
# without the shared driver context.
#
# Both cards must be the SAME MODEL. Two different GPUs would confound the
# comparison with a hardware difference, which is the thing this design exists
# to avoid.
GPU32=${GPU32:-0}
GPU64=${GPU64:-1}
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$NGPU" -lt 2 ]; then
  echo "ERROR: this script needs two GPUs; found $NGPU." >&2
  echo "Run it on a two-card box, or set GPU32=0 GPU64=0 to accept the" >&2
  echo "shared-card design that deadlocked on 2026-10-09." >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name --format=csv,noheader | tee "$OUT/gpus.txt"
if [ "$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | wc -l)" -ne 1 ]; then
  echo "ERROR: the two cards are different models; that confounds the arms." >&2
  exit 1
fi

common="--considered 32 --keep-every 50 --games 256 --concurrency 256 \
        --max-plies 400 --steps 250 --batch 512 --window 8 --lr 5e-4 \
        --blocks 6 --channels 96 --generations 100000 --max-hours $HOURS"

echo "=== sims A/B, started $(date) ===" | tee "$OUT/sims-ab.log"
echo "sims32 on GPU$GPU32, sims64 on GPU$GPU64, ${HOURS}h wall clock, considered=32" | tee -a "$OUT/sims-ab.log"

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=$GPU32 .venv/bin/chess-train $common --sims 32 \
  --out "$OUT/ab-sims32" > "$OUT/ab-sims32.log" 2>&1 &
P32=$!
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=$GPU64 .venv/bin/chess-train $common --sims 64 \
  --out "$OUT/ab-sims64" > "$OUT/ab-sims64.log" 2>&1 &
P64=$!
echo "arms running: sims32 pid $P32, sims64 pid $P64" | tee -a "$OUT/sims-ab.log"

# A CUDA deadlock does not kill the process. On 2026-10-09 one arm wedged in the
# driver (state D, os_acquire_rwlock_read) and sat there 9.7h while the other ran
# on: pgrep said 2/2 and the GPU showed 74% because the survivor was using it.
# Liveness tells you nothing; progress does. If an arm stops advancing, abort
# BOTH -- a wall-clock-matched comparison is already dead at that point, and
# burning another nine hours only makes the bill worse.
STALL_MIN=${STALL_MIN:-20}
watchdog () {
  local last32=-1 last64=-1 since=0
  while kill -0 "$P32" 2>/dev/null && kill -0 "$P64" 2>/dev/null; do
    sleep 300
    local g32 g64
    g32=$(grep -cE '^ +[0-9]+ +256' "$OUT/ab-sims32.log" 2>/dev/null || echo 0)
    g64=$(grep -cE '^ +[0-9]+ +256' "$OUT/ab-sims64.log" 2>/dev/null || echo 0)
    if [ "$g32" -eq "$last32" ] || [ "$g64" -eq "$last64" ]; then
      since=$((since + 5))
      if [ "$since" -ge "$STALL_MIN" ]; then
        {
          echo "WATCHDOG: no progress for ${since}min (sims32 $g32 gens, sims64 $g64 gens)"
          echo "WATCHDOG: an arm has stalled -- the matched comparison is void. Aborting both."
          echo "WATCHDOG: state: $(ps -o stat= -p $P32 2>/dev/null) / $(ps -o stat= -p $P64 2>/dev/null) (D = wedged in the driver)"
        } | tee -a "$OUT/sims-ab.log"
        kill -9 "$P32" "$P64" 2>/dev/null
        return
      fi
    else
      since=0
    fi
    last32=$g32; last64=$g64
  done
}
watchdog &
WD=$!

wait $P32 $P64
kill "$WD" 2>/dev/null

echo "=== both arms finished $(date) ===" | tee -a "$OUT/sims-ab.log"
for a in ab-sims32 ab-sims64; do
  g=$(ls "$OUT/$a"/gen*.pt 2>/dev/null | wc -l)
  last=$(.venv/bin/python -c "
import json;h=json.load(open('$OUT/$a/history.json'));print(f\"{len(h)} generations, {h[-1]['elapsed_total']/3600:.1f}h\")" 2>/dev/null)
  echo "$a: $last, $g checkpoints kept" | tee -a "$OUT/sims-ab.log"
done
echo "next: booked head-to-head between the two, plus equal-data and" | tee -a "$OUT/sims-ab.log"
echo "      equal-wall-clock curves from the kept checkpoints" | tee -a "$OUT/sims-ab.log"
