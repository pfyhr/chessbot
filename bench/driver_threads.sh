#!/bin/zsh
# Does threading the self-play driver at game granularity pay? -- Jupiter.
#
# The driver is single-threaded on purpose. On 2026-10-06 the loops inside
# next_batch were handed to rayon per slot and measured SLOWER: 4.8s -> 5.4s at
# 4 threads (86b661a, reverted in 2131d58). A call carries ~5ms of work, so
# per-slot dispatch cost more than the work it handed out.
#
# This measures a different thing. Slots are now sharded into contiguous blocks,
# one per thread, each owning whole games -- its own trees, its own trajectories,
# its own rows of the batch. A call dispatches `threads` tasks, not
# `concurrency` of them, on a pool that is built once.
#
# WHY IT MATTERS, and it is not Jupiter. Here the GPU is ~87 of 97 seconds, so
# halving the driver buys a few percent. On a rented 4090 the predicted split is
# 18.7s driver against 13.6s GPU -- 58% of the rental is one core doing tree
# descent (docs/rented-training-plan.md). The number that transfers between
# machines is DRIVER SECONDS PER GENERATION, not wall clock, so phase 2 reports
# it explicitly.
#
# DO NOT RUN WHILE TRAINING OR AN FEA JOB IS LIVE. A timing measurement taken
# under contention measures the contention. Jupiter's CPU is shared with
# srm-studio-34; agree a window first.
set -u
# The repo root, from where this script lives, so a clone anywhere works.
R=$(cd "$(dirname "$0")/.." && pwd)
OUT=$R/runs/driver-threads
# Budget: phase 1 ~4 min, phase 2 ~5 min, phase 3 ~40 min. About 50 minutes in
# all at these defaults. GENS=3 BLOCKS=1 cuts phase 3 to ten and still gives a
# paired comparison, if the window is shorter than agreed.
GENS=${GENS:-5}        # per block in phase 3; gen 0 pays compile and is dropped
BLOCKS=${BLOCKS:-2}    # A-B pairs
ARM_B=${ARM_B:-4}      # threads for the B arm
SWEEP=${SWEEP:-"1 2 4 8"}
mkdir -p $OUT
L=$OUT/report.log

cd $R || exit 1
print "=== driver threading, started $(date) ===" | tee -a $L
print "host $(hostname)  $(sysctl -n machdep.cpu.brand_string)  $(sysctl -n hw.ncpu) cores" | tee -a $L
print "load at start: $(uptime | sed 's/.*load averages: //')" | tee -a $L

if pgrep -f "[c]hess-train" > /dev/null; then
  print "REFUSING TO RUN: training is live." | tee -a $L
  exit 1
fi
LOAD1=$(uptime | awk -F'load averages: ' '{print $2}' | awk '{print $1}')
if (( ${LOAD1%.*} > 2 )); then
  print "" | tee -a $L
  print "WARNING: 1-minute load is $LOAD1 before anything of ours has started." | tee -a $L
  print "Something else owns these cores. Stop, or every number below is noise." | tee -a $L
  [ "${FORCE:-0}" = "1" ] || exit 1
fi

print "" | tee -a $L
print -r -- "--- phase 1: driver ceiling, network excluded ---" | tee -a $L
print "drivercost feeds an instant evaluator, so this is the driver alone." | tee -a $L
nice -n 5 cargo build --release -p az-core --bin drivercost 2>&1 | tail -1 | tee -a $L
for t in ${=SWEEP}; do
  nice -n 5 ./target/release/drivercost --sims 32 --concurrency 512 --games 2048 --threads $t \
    > $OUT/drivercost-$t.txt 2>&1
  print "  threads $t: $(grep 'evals/sec (' $OUT/drivercost-$t.txt | awk '{print $NF}') evals/sec" | tee -a $L
done
$R/.venv/bin/python - "$OUT" ${=SWEEP} <<'PY' | tee -a $L
import re, sys
out, ts = sys.argv[1], sys.argv[2:]
rates = {}
for t in ts:
    txt = open(f"{out}/drivercost-{t}.txt").read()
    m = re.search(r"evals/sec \(\d+ threads?\)\s+([\d,]+)", txt)
    if m:
        rates[int(t)] = float(m.group(1).replace(",", ""))
if rates:
    base = rates[min(rates)]
    print(f"\n  {'threads':>8s} {'evals/sec':>12s} {'speedup':>9s} {'efficiency':>11s}")
    for t in sorted(rates):
        print(f"  {t:>8d} {rates[t]:>12,.0f} {rates[t]/base:>8.2f}x {rates[t]/base/t:>10.0%}")
    print("\n  2026-08-20 measured 292,646 evals/sec on one thread of this machine.")
    best = max(rates, key=rates.get)
    if rates[best] / base < 1.3:
        print("  THREADING DOES NOT PAY at this granularity either. Say so and stop;")
        print("  the rented-GPU plan then has to assume a single-threaded driver.")
PY

print "" | tee -a $L
print -r -- "--- phase 2: driver seconds per generation ---" | tee -a $L
print "No torch.compile here on purpose: generation_profile feeds a VARIABLE" | tee -a $L
print "batch size and compile would recompile for every shape. Compile only" | tee -a $L
print "touches the GPU half anyway, and phase 3 measures it with bucketing on." | tee -a $L
for t in 1 2 $ARM_B; do
  nice -n 5 $R/.venv/bin/python bench/generation_profile.py \
      --games 128 --concurrency 128 --sims 32 --considered 32 --steps 100 \
      --driver-threads $t > $OUT/profile-$t.txt 2>&1
  print "  threads $t:" | tee -a $L
  grep -E "driver seconds|GPU-bound|generation total" $OUT/profile-$t.txt | sed 's/^/    /' | tee -a $L
done

print "" | tee -a $L
print -r -- "--- phase 3: real s/gen, compile on, alternating blocks ---" | tee -a $L
print "Alternating A-B-A-B and sequential, as in compile_ab.sh: a laptop warms" | tee -a $L
print "over an hour, and running the arms concurrently would halve both." | tee -a $L

block () {  # label, threads, index
  local label=$1 threads=$2 idx=$3
  local d=$OUT/${label}-$idx
  rm -rf $d; mkdir -p $d
  nice -n 5 $R/.venv/bin/chess-train \
    --init $R/runs/long-oct/gen2600.pt --out $d --compile \
    --driver-threads $threads \
    --generations $GENS --max-hours 1 \
    --games 256 --concurrency 256 --sims 32 --considered 32 \
    --max-plies 400 --steps 250 --batch 512 --lr 5e-4 --window 8 \
    --blocks 6 --channels 96 --eval-every 100000 --keep-every 100000 \
    > $d/train.log 2>&1
  # history.json, never train.log: Python buffers stdout to a file and it lies.
  # Average `seconds` over the clean generations only: generation 0 pays
  # torch.compile and startup, and `evaluated` ones carry a tactics suite and
  # two matches that have nothing to do with the driver. history.json, never
  # train.log -- Python buffers stdout to a file and it lies.
  $R/.venv/bin/python -c "
import json
h=json.load(open('$d/history.json'))
clean=[g for g in h if g['gen'] > 0 and not g['evaluated']]
if not clean:
    print('  $label-$idx  NO CLEAN GENERATIONS -- raise GENS'); raise SystemExit
rate=sum(g['seconds'] for g in clean)/len(clean)
plies=sum(g['mean_plies'] for g in clean)/len(clean)
print(f'  $label-$idx  {rate:6.1f}s/gen over {len(clean)} clean gens, mean_plies {plies:.1f}')" | tee -a $L
}

for i in $(seq 1 $BLOCKS); do
  block "threads-1"       1      $i
  block "threads-$ARM_B"  $ARM_B $i
done

print "" | tee -a $L
$R/.venv/bin/python - "$L" "$ARM_B" <<'PY' | tee -a $L
import re, sys
log, armb = sys.argv[1], sys.argv[2]
a, b = [], []
for ln in open(log):
    m = re.search(r"threads-(\d+)-\d+\s+([\d.]+)s/gen", ln)
    if m:
        (a if m.group(1) == "1" else b).append(float(m.group(2)))
if a and b:
    x, y = sum(a)/len(a), sum(b)/len(b)
    print(f"  1 thread {x:6.1f}s/gen   {armb} threads {y:6.1f}s/gen   speedup {x/y:.2f}x")
    print(f"  Do NOT compare this to long-oct's 97.0s/gen: every generation there")
    print(f"  has evaluated:true, so 97.0 includes a ~12s probe suite that these")
    print(f"  blocks exclude by design. The 1-thread arm above is the control.")
    print(f"  Jupiter is ~90% GPU, so a small number here is EXPECTED and is not")
    print(f"  the case for the change. Read phase 2's driver seconds instead:")
    print(f"  that is what a rented 4090 would pay, where the split is 58/42.")
PY
print "load at end: $(uptime | sed 's/.*load averages: //')" | tee -a $L
print "=== COMPLETE $(date) ===" | tee -a $L
