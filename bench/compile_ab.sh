#!/bin/zsh
# Is torch.compile why generations got 1.5x faster? -- run AFTER runs/long-oct.
#
# runs/long-oct averages 95s/generation. The weekend run of 2026-09-20, same
# config and same mean game length (~130 plies), averaged 141s and was FLAT at
# 138-143s across all 1,372 of its generations -- so the speedup is not a
# warm-up or game-length effect. It is software: a newer torch, torch.compile
# (on by default since), and bridge work in between. This isolates the compile
# half of that.
#
# DO NOT RUN WHILE TRAINING IS LIVE. It competes for the same 8 P-cores and the
# GPU, and a timing measurement taken under contention measures the contention.
#
# Design: alternating A-B-A-B blocks, SEQUENTIALLY. Not concurrent -- for a
# throughput comparison, running both arms at once halves both and tells you
# nothing. Alternating blocks keep thermal drift (a laptop warms over an hour)
# symmetric between the arms instead of loading it onto whichever ran second.
set -u
# The repo root, from where this script lives, so a clone anywhere works.
R=$(cd "$(dirname "$0")/.." && pwd)
OUT=$R/runs/compile-ab
GENS=${GENS:-6}      # per block; the first is discarded as warm-up
BLOCKS=${BLOCKS:-2}  # A-B pairs
mkdir -p $OUT
L=$OUT/report.log

cd $R || exit 1
print "=== torch.compile A/B, started $(date) ===" | tee -a $L
$R/.venv/bin/python -c "import torch,sys;print(f'torch {torch.__version__}  python {sys.version.split()[0]}  mps {torch.backends.mps.is_available()}')" | tee -a $L
print "$GENS generations per block, $BLOCKS alternating A-B pairs, sequential" | tee -a $L
print "" | tee -a $L

if pgrep -f "[c]hess-train" > /dev/null; then
  print "REFUSING TO RUN: training is live. A timing test under contention" | tee -a $L
  print "measures the contention, not the thing being tested." | tee -a $L
  exit 1
fi

block () {  # label, extra-flag, index
  local label=$1 flag=$2 idx=$3
  local d=$OUT/${label}-$idx
  rm -rf $d; mkdir -p $d
  $R/.venv/bin/chess-train \
    --init $R/runs/weekend/gen1371.pt --out $d $flag \
    --generations $GENS --max-hours 2 \
    --games 256 --concurrency 256 --sims 32 --considered 32 \
    --max-plies 400 --steps 250 --batch 512 --lr 5e-4 --window 8 \
    --blocks 6 --channels 96 --eval-every 1 --keep-every 100000 \
    > $d/train.log 2>&1
  # history.json, never train.log: Python buffers stdout to a file and it lies.
  $R/.venv/bin/python -c "
import json
h=json.load(open('$d/history.json'))
warm=h[1:]           # discard generation 0, which pays compile and startup
rate=(warm[-1]['elapsed_total']-warm[0]['elapsed_total'])/max(1,len(warm)-1)
plies=sum(x.get('mean_plies',0) for x in warm)/len(warm)
print(f'  $label-$idx  {rate:6.1f}s/gen over {len(warm)} gens, mean_plies {plies:.1f}, gen0 {h[0][\"seconds\"]:.0f}s')" | tee -a $L
}

for i in $(seq 1 $BLOCKS); do
  block "compile-on"  "--compile"    $i
  block "compile-off" "--no-compile" $i
done

print "" | tee -a $L
$R/.venv/bin/python -c "
import re
on, off = [], []
for ln in open('$L'):
    m = re.search(r'(compile-\w+)-\d+\s+([\d.]+)s/gen', ln)
    if m: (on if m.group(1)=='compile-on' else off).append(float(m.group(2)))
if on and off:
    a, b = sum(on)/len(on), sum(off)/len(off)
    print(f'  compile ON  {a:6.1f}s/gen   OFF {b:6.1f}s/gen   speedup {b/a:.2f}x')
    print(f'  long-oct measured 95s/gen; the weekend run 141s (ratio 1.48x)')
    print(f'  -> compile explains {(b/a-1)/(141/95-1)*100:.0f}% of the gap' if b>a else
          '  -> compile does NOT explain the speedup; look at torch version or the bridge')
" | tee -a $L
print "=== COMPLETE $(date) ===" | tee -a $L
