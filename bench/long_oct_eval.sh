#!/bin/zsh
# What did 72 hours of training actually buy? -- runs/long-oct, Oct 2026.
#
# long-oct ran 2,601 generations from runs/weekend/gen1371.pt. Its own probes
# are discouraging and need adjudicating by games, not by proxies:
#
#   vs_random   0.799 (gens 0-199) -> 0.845 (gens 2400-2599), inside the noise
#   vs_prev     pinned at 0.500 for the entire run
#   policy_loss 2.335 -> 2.244, a 3.9% improvement
#   value_loss  0.366 -> 0.384, WORSE
#   mate/defend/tactics  0.023/0.390/0.207 -> 0.012/0.293/0.152, all down
#
# Four metrics have given this project confidently wrong answers before, so
# none of the above decides anything. Head-to-head games do.
#
# Bounds [0,50]: the [0,10] validation was correct but useless at our scale --
# 244 games for a +197 effect, 800 inconclusive on a true null. 50 Elo is the
# "worth changing what we do" threshold used by bench/sprt_bounds.sh.
#
# SPRT IS DIRECTIONAL: engine A is the candidate. H1 = "A is better by >=50".
set -u
R=/Users/pontus/repos/chessbot
FC=/Users/pontus/repos/fastchess/fastchess
OUT=$R/runs/long-oct-eval
mkdir -p $OUT
L=$OUT/report.log
# 4 P-cores: srm-studio-34 runs FEA at 6 workers alongside this.
CONC=${CONC:-4}
E1=${E1:-50} ALPHA=${ALPHA:-0.05} BETA=${BETA:-0.05}
NODES=${NODES:-128} RA=${RA:-32}
CAP=${CAP:-600}

cd $R || exit 1
# fastchess autosaves its state to ./config.json; run it from a scratch dir so
# it stops dirtying the repo root. All paths below are absolute.
FCWORK=${TMPDIR:-/tmp}/fastchess-cwd
mkdir -p $FCWORK
print "=== long-oct evaluation, started $(date) ===" | tee -a $L
print "SPRT [0,$E1] alpha=$ALPHA beta=$BETA logistic, cap=$((CAP*2)) games" | tee -a $L
print "$NODES nodes, RootActions=$RA, concurrency=$CONC (sharing with FEA)" | tee -a $L
print "" | tee -a $L

duel () {  # label, runA, genA, runB, genB, question
  local label=$1 ra=$2 ga=$3 rb=$4 gb=$5 q=$6
  local raw=$OUT/${label}.raw
  local t0=$SECONDS
  ( cd $FCWORK && $FC \
    -engine cmd=$R/.venv/bin/chess-uci args="--run $R/runs/$ra" \
      name=A option.Generation=$ga option.RootActions=$RA nodes=$NODES \
    -engine cmd=$R/.venv/bin/chess-uci args="--run $R/runs/$rb" \
      name=B option.Generation=$gb option.RootActions=$RA nodes=$NODES \
    -sprt elo0=0 elo1=$E1 alpha=$ALPHA beta=$BETA model=logistic \
    -rounds $CAP -games 2 -concurrency $CONC -srand $((RANDOM)) \
    -openings file=$R/bench/openings-4ply.epd format=epd order=random \
    -log file=/dev/null > $raw 2>&1 )
  print "$label  ($ra/gen$ga  vs  $rb/gen$gb)" | tee -a $L
  print "    question: $q" | tee -a $L
  print "    $(grep -E '^Games:' $raw | tail -1)" | tee -a $L
  print "    $(grep -E '^Elo:' $raw | tail -1)" | tee -a $L
  print "    $(grep -iE 'LLR' $raw | tail -1)" | tee -a $L
  print "    verdict: $(grep -iE 'H1 was accepted|H0 was accepted|SPRT.*accepted' $raw | tail -1 || true)" | tee -a $L
  print "    took $((SECONDS-t0))s" | tee -a $L
  print "" | tee -a $L
}

# positive Elo = engine A won.
duel whole-run  long-oct 2600 weekend  1371 \
     "did 72h and 2601 generations beat the checkpoint it started from?"
duel second-half long-oct 2600 long-oct 1300 \
     "did the second 36h add anything to the first?"
duel first-half  long-oct 1300 weekend  1371 \
     "did the first 36h add anything at all?"

print "=== COMPLETE $(date) ===" | tee -a $L
