#!/bin/zsh
# sims 32 vs sims 64 at EQUAL WALL CLOCK -- the question the 9 Oct A/B was built
# to answer, salvaged. M2, Oct 2026.
#
# This was declared unrecoverable, and for the run as a whole it is: the sims32
# arm wedged in the NVIDIA driver at 6.04h, after which sims64 had the GPU to
# itself and its later generations are faster than contention would have allowed.
# Any comparison involving those generations is contaminated.
#
# But BELOW 6.04h both arms were running concurrently, so contention was
# symmetric by construction -- which is exactly the property the design needed.
# Every checkpoint pair under that line is a valid equal-wall-clock comparison.
# The deadlock cost the top of the curve, not the curve.
#
# sims64 costs 1.53x per generation, so at a matched hour it is always the
# FEWER-generation side. Equal data (bench/sims_equal_data.sh) found it ahead by
# -113 +/-30 Elo late and BEHIND by +101 +/-37 early. The question here is
# whether the late advantage is big enough to pay for the generations it costs.
set -u
R=/Users/pontus/repos/chessbot
FC=${FC:-/Users/pontus/repos/fastchess/fastchess}
[ -x "$FC" ] || FC=$HOME/fastchess/fastchess
A=$R/runs/ab-salvage/ab-sims32
B=$R/runs/ab-salvage/ab-sims64
OUT=$R/runs/sims-equal-wallclock
mkdir -p $OUT
L=$OUT/report.log
RAW=$OUT/raw.log
NODES=${NODES:-128}
CONC=${CONC:-$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo 4)}
ROUNDS=${ROUNDS:-75}

cd $R || exit 1
print "=== sims32 vs sims64 at equal WALL CLOCK, started $(date) ===" | tee -a $L
print "both sides $NODES nodes, RootActions 32; pairs matched on elapsed_total" | tee -a $L
print "all pairs below the 6.04h deadlock, so both arms shared the GPU" | tee -a $L
print "positive Elo = sims32 (the cheaper arm) won" | tee -a $L
print "" | tee -a $L

duel () {  # g32, g64, hours-label, seed
  local g32=$1 g64=$2 hrs=$3 seed=$4
  local t0=$SECONDS
  $FC \
    -engine cmd=$R/.venv/bin/chess-uci args="--run $A" \
      name=sims32 option.Generation=$g32 option.RootActions=32 nodes=$NODES \
    -engine cmd=$R/.venv/bin/chess-uci args="--run $B" \
      name=sims64 option.Generation=$g64 option.RootActions=32 nodes=$NODES \
    -rounds $ROUNDS -games 2 -concurrency $CONC -srand $seed \
    -openings file=$R/bench/openings-4ply.epd format=epd order=random \
    -log file=/dev/null >> $RAW 2>&1
  local e g
  e=$(grep -E "^Elo:" $RAW | tail -1)
  g=$(grep -E "^Games:" $RAW | tail -1)
  print "$hrs  s32-gen$g32 vs s64-gen$g64  ::  $e  ::  $g  ::  $((SECONDS - t0))s" | tee -a $L
}

# The three best-matched pairs among saved checkpoints. The residual mismatch is
# noted because it is not symmetric: at ~5h sims64 has 3% MORE time, at the two
# earlier rungs sims32 has 7-8% more. So the early rungs flatter sims32 slightly
# and the late rung flatters sims64 slightly -- which happens to bias each rung
# AGAINST the arm the equal-data curve says should win there.
duel 350 200 "~5.0h (4.95 vs 5.10, s64 +3%)" 20261024
duel 200 100 "~2.8h (2.95 vs 2.73, s32 +8%)" 20261025
duel 100  50 "~1.5h (1.56 vs 1.45, s32 +7%)" 20261026

print "" | tee -a $L
print "timeouts: $(grep -c 'loses on time' $RAW 2>/dev/null || print 0)" | tee -a $L
print "=== COMPLETE $(date) ===" | tee -a $L
