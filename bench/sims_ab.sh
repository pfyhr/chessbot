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
#   * both arms run CONCURRENTLY on one GPU, so contention is identical. An arm
#     that runs alone cannot be compared against one that shared -- that is what
#     made cap-xl uninterpretable.
#   * wall-clock matched, not generation matched. The question is what a budget
#     buys per hour, and the 64-sims arm pays for its search in games forgone.
#   * considered=32 on both, stated explicitly: it is worth ~+102 Elo and was
#     never recorded in any past run's log.
#   * keep-every 50: a checkpoint is ~126 MB, every generation would be 113 GB,
#     and even every 25th is 11.6 GB against the 16 GB this box has free.
#
# Probed on an RTX 5080: 60.5s and 98.8s per generation, so 24h gives roughly
# 1,428 and 874 generations against the 575/386 the capacity A/B worked from.
set -u
R=$HOME/chessbot
OUT=$R/runs
HOURS=${HOURS:-24}
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd "$R" || exit 1

common="--considered 32 --keep-every 50 --games 256 --concurrency 256 \
        --max-plies 400 --steps 250 --batch 512 --window 8 --lr 5e-4 \
        --blocks 6 --channels 96 --generations 100000 --max-hours $HOURS"

echo "=== sims A/B, started $(date) ===" | tee "$OUT/sims-ab.log"
echo "both arms concurrent on one GPU, ${HOURS}h wall clock, considered=32" | tee -a "$OUT/sims-ab.log"

# shellcheck disable=SC2086
.venv/bin/chess-train $common --sims 32 --out "$OUT/ab-sims32" > "$OUT/ab-sims32.log" 2>&1 &
P32=$!
# shellcheck disable=SC2086
.venv/bin/chess-train $common --sims 64 --out "$OUT/ab-sims64" > "$OUT/ab-sims64.log" 2>&1 &
P64=$!
echo "arms running: sims32 pid $P32, sims64 pid $P64" | tee -a "$OUT/sims-ab.log"
wait $P32 $P64

echo "=== both arms finished $(date) ===" | tee -a "$OUT/sims-ab.log"
for a in ab-sims32 ab-sims64; do
  g=$(ls "$OUT/$a"/gen*.pt 2>/dev/null | wc -l)
  last=$(.venv/bin/python -c "
import json;h=json.load(open('$OUT/$a/history.json'));print(f\"{len(h)} generations, {h[-1]['elapsed_total']/3600:.1f}h\")" 2>/dev/null)
  echo "$a: $last, $g checkpoints kept" | tee -a "$OUT/sims-ab.log"
done
echo "next: booked head-to-head between the two, plus equal-data and" | tee -a "$OUT/sims-ab.log"
echo "      equal-wall-clock curves from the kept checkpoints" | tee -a "$OUT/sims-ab.log"
