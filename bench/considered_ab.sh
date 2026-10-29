#!/bin/zsh
# Does a narrower root train better at the same budget? -- Jupiter, Oct 2026.
#
# At --sims 32 --considered 32, mcts.rs:522 gives visits_each = max(32/160,1) = 1,
# so phase one spends the whole budget on one visit per root action and
# sequential halving never runs. Measured depth (gen695): 1.57 plies.
# At --considered 8 all three halving phases fit: depth 2.58, and 26.8
# evaluations per move against 31.8 -- 64% deeper and 16% cheaper.
#
# BEFORE RUNNING THIS, do steps 1-2 of docs/root-width-plan.md. They measure
# target quality directly in about an hour and may remove the need for this
# script entirely. At ~0.073 Elo/generation, two arms need ~1500 generations
# each before the gap is large enough to resolve -- days of compute.
#
# Whether that trains better is a separate question, and the only thing that
# answers it is games. long-oct's probes said the run gained nothing; the games
# said +249 Elo. Probes here are secondary evidence, nothing more.
#
# SEQUENTIAL, not concurrent: one GPU, and two CUDA/MPS trainers on one device
# have wedged this project before. Cost per generation is read per-arm from
# history.json, so equal-wall-clock is reconstructed rather than staged.
set -u
# The repo root, from where this script lives, so a clone anywhere works.
R=$(cd "$(dirname "$0")/.." && pwd)
OUT=$R/runs/considered-ab
GENS=${GENS:-300}
INIT=${INIT:-$R/runs/long-oct/gen2600.pt}
mkdir -p $OUT
L=$OUT/report.log

cd $R || exit 1
if pgrep -f "[c]hess-train" > /dev/null; then
  print "REFUSING: training already live." | tee -a $L; exit 1
fi
print "=== considered A/B, started $(date) ===" | tee -a $L
print "init $INIT, $GENS generations per arm, --sims 32, sequential" | tee -a $L
$R/.venv/bin/python -c "
import chessbot_core as cc
print('  bridge counters present:', hasattr(cc,'hot_profile'))" | tee -a $L
print "" | tee -a $L

arm () {  # label, considered
  local label=$1 c=$2
  local d=$OUT/$label
  rm -rf $d; mkdir -p $d
  print "--- $label (--considered $c) started $(date '+%H:%M') ---" | tee -a $L
  $R/.venv/bin/chess-train \
    --init $INIT --out $d \
    --generations $GENS --max-hours 24 \
    --games 256 --concurrency 256 --sims 32 --considered $c \
    --max-plies 400 --steps 250 --batch 512 --lr 5e-4 --window 8 \
    --blocks 6 --channels 96 --eval-every 1 --keep-every 50 \
    > $d/train.log 2>&1
  $R/.venv/bin/python -c "
import json
h=json.load(open('$d/history.json'))
w=h[1:]
rate=(w[-1]['elapsed_total']-w[0]['elapsed_total'])/max(1,len(w)-1)
f=lambda k,s: sum(x.get(k,0) or 0 for x in s)/len(s)
print(f'  $label: {len(h)} gens, {rate:.1f}s/gen, {f(\"evaluations\",w):.0f} evals/gen')
print(f'    first 50: mate {f(\"mate\",h[:50]):.3f} defend {f(\"defend\",h[:50]):.3f} ploss {f(\"policy_loss\",h[:50]):.3f} vloss {f(\"value_loss\",h[:50]):.3f}')
print(f'    last  50: mate {f(\"mate\",h[-50:]):.3f} defend {f(\"defend\",h[-50:]):.3f} ploss {f(\"policy_loss\",h[-50:]):.3f} vloss {f(\"value_loss\",h[-50:]):.3f}')" | tee -a $L
  print "" | tee -a $L
}

# Sequential Halving needs n >= m*log2(m) for its per-phase visit floor to be
# at least 1 (Danihelka et al. 2022, Fig. 1). At n=32: m=8 needs 24 and fits,
# m=16 needs 64 and completes 3 of 4 phases, m=32 needs 160 and completes none.
# Run the middle arm too -- it is the paper's default constant, and it is the
# one our own Config::default() already carries.
arm narrow-8   8
arm middle-16 16
arm wide-32   32

print "Arms trained. Judge by games, not by the lines above:" | tee -a $L
print "  bench/considered_match.sh" | tee -a $L
print "=== COMPLETE $(date) ===" | tee -a $L
