#!/bin/zsh
# Does SPRT give the right answer, and how many games does it cost? -- M2.
#
# Every Elo figure in this project is a fixed-length match read at face value,
# which has burned us twice: ra8 came in at +58 +/-34 and shrank to +41 pooled
# on replication, because the number reported was the largest of three noisy
# estimates. SPRT instead runs until the evidence crosses a decision boundary,
# so it controls the error rate explicitly rather than after the fact.
#
# Before adopting it, test it on two cases whose answers we already know:
#
#   A  POSITIVE CONTROL. Same network, 256 nodes against 64. The node ladder
#      puts this near +190 Elo (128-over-64 is +100 +/-39, 256-over-128 is
#      +90 +/-44). SPRT must accept H1, and should do it quickly.
#   B  NULL CONTROL. The same network against itself at the same budget, only
#      the seed differs. True Elo is 0. SPRT must NOT accept H1 -- this is the
#      false-positive check, and it is the half that a point estimate skips.
#
# H0: the change is worth 0 Elo.  H1: it is worth at least 10.
set -u
# The repo root, from where this script lives, so a clone anywhere works.
R=$(cd "$(dirname "$0")/.." && pwd)
FC=${FC:-$(dirname "$R")/fastchess/fastchess}
OUT=$R/runs/sprt-validate
mkdir -p $OUT
L=$OUT/report.log
CONC=${CONC:-$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo 8)}
ELO0=${ELO0:-0} ELO1=${ELO1:-10} ALPHA=${ALPHA:-0.05} BETA=${BETA:-0.05}
MODEL=${MODEL:-logistic}   # logistic = the Elo scale every past result used
CAP=${CAP:-400}            # rounds; the null test would otherwise run forever

cd $R || exit 1
# fastchess autosaves its state to ./config.json; run it from a scratch dir so
# it stops dirtying the repo root. All paths below are absolute.
FCWORK=${TMPDIR:-/tmp}/fastchess-cwd
mkdir -p $FCWORK
print "=== SPRT validation, started $(date) ===" | tee -a $L
print "H0=$ELO0 H1=$ELO1 alpha=$ALPHA beta=$BETA model=$MODEL cap=$((CAP*2)) games conc=$CONC" | tee -a $L
print "" | tee -a $L

run () {  # label, nodesA, nodesB, seed, expected
  local label=$1 na=$2 nb=$3 seed=$4 expect=$5
  local raw=$OUT/${label}.raw
  local t0=$SECONDS
  ( cd $FCWORK && $FC \
    -engine cmd=$R/.venv/bin/chess-uci args="--run $R/runs/w-long" \
      name=A option.Generation=695 option.RootActions=32 nodes=$na \
    -engine cmd=$R/.venv/bin/chess-uci args="--run $R/runs/w-long" \
      name=B option.Generation=695 option.RootActions=32 nodes=$nb \
    -sprt elo0=$ELO0 elo1=$ELO1 alpha=$ALPHA beta=$BETA model=$MODEL \
    -rounds $CAP -games 2 -concurrency $CONC -srand $seed \
    -openings file=$R/bench/openings-4ply.epd format=epd order=random \
    -log file=/dev/null > $raw 2>&1 )
  local games llr res elo
  games=$(grep -E "^Games:" $raw | tail -1)
  llr=$(grep -iE "LLR" $raw | tail -1)
  elo=$(grep -E "^Elo:" $raw | tail -1)
  res=$(grep -iE "H1 was accepted|H0 was accepted|SPRT.*accepted" $raw | tail -1)
  print "$label  (A=$na nodes vs B=$nb nodes, expect: $expect)" | tee -a $L
  print "    $games" | tee -a $L
  print "    $elo" | tee -a $L
  print "    $llr" | tee -a $L
  print "    verdict: ${res:-no boundary crossed within cap}" | tee -a $L
  print "    took $((SECONDS-t0))s" | tee -a $L
  print "" | tee -a $L
}

run "A-positive" 64 256 20261101 "H1 accepted — B is better"
run "B-null"    128 128 20261102 "H1 NOT accepted — no real difference"

print "=== COMPLETE $(date) ===" | tee -a $L
