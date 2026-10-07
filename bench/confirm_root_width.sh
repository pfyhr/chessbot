#!/bin/zsh
# Does RootActions 8 survive a second look? -- M2, overnight 7-8 Oct 2026.
#
# The 6-7 Oct study found ra8 beating the ra32 default by +58 +/-34 at a fixed
# 128 nodes. Two things keep that from being adoptable:
#
#   1. Three comparisons were run that night (ra4, ra8, ra16 against ra32), so a
#      single false positive at 95% carries roughly 14% risk. REPLICATE repeats
#      the ra8 cell alone, on a different opening draw, which is an independent
#      test rather than a re-reading of the same games.
#
#   2. It is one budget. GENERALITY repeats it at 512 nodes, where depth is
#      already cheap -- if narrowing only pays when the tree is shallow, the
#      margin should shrink here, and the setting belongs per node count rather
#      than as a global default.
#
# Both sides are gen695 throughout, so this is search only and none of the
# short-run training caveats apply. Fixed nodes, so a busy laptop cannot distort
# it. Results append as they land.
set -u
R=/Users/pontus/repos/chessbot
FC=/Users/pontus/repos/fastchess/fastchess
OUT=$R/runs/confirm-ra8
mkdir -p $OUT
L=$OUT/report.log
RAW=$OUT/raw.log
ROUNDS=150        # 300 games per pairing, ~ +/-34 Elo
CONC=4            # as every past M2 match here; calibrated at 136 games / 42 min

cd $R || exit 1
print "=== RootActions 8 confirmation, started $(date) ===" | tee -a $L
print "both sides runs/w-long gen695; only RootActions differs" | tee -a $L
print "" | tee -a $L

duel () {  # label, nodes, seed
  local label=$1 nodes=$2 seed=$3
  local t0=$SECONDS
  $FC \
    -engine cmd=$R/.venv/bin/chess-uci args="--run $R/runs/w-long" \
      name=ra8  option.Generation=695 option.RootActions=8  nodes=$nodes \
    -engine cmd=$R/.venv/bin/chess-uci args="--run $R/runs/w-long" \
      name=ra32 option.Generation=695 option.RootActions=32 nodes=$nodes \
    -rounds $ROUNDS -games 2 -concurrency $CONC -srand $seed \
    -openings file=$R/bench/openings-4ply.epd format=epd order=random \
    -log file=/dev/null >> $RAW 2>&1
  local e g
  e=$(grep -E "^Elo:" $RAW | tail -1)
  g=$(grep -E "^Games:" $RAW | tail -1)
  print "$label  ::  $e  ::  $g  ::  $((SECONDS - t0))s" | tee -a $L
}

# Cheap one first, so a short night still answers the more important question.
duel "REPLICATE  ra8 over ra32 @ 128 nodes" 128 20261007
duel "GENERALITY ra8 over ra32 @ 512 nodes" 512 20261008

print "" | tee -a $L
print "timeouts: $(grep -c 'loses on time' $RAW 2>/dev/null || print 0)" | tee -a $L
print "=== COMPLETE $(date) ===" | tee -a $L
