#!/bin/zsh
# sims 32 vs sims 64 in self-play, compared at EQUAL DATA -- M2, Oct 2026.
#
# The wall-clock A/B this was meant to settle is dead: on 2026-10-09 the sims32
# arm wedged inside the NVIDIA driver after 6.04h while the sims64 arm ran on,
# so the matched wall clock the design rested on no longer exists. See
# [[two-cuda-processes-one-gpu-deadlocks]].
#
# What survived is 9 matched checkpoints per arm, gen 0 to 400 in steps of 50.
# Both arms played 256 self-play games per generation, so at any matched
# generation the two networks have seen the SAME NUMBER OF GAMES. That makes an
# equal-data comparison still exact, and it answers the half of the question
# that matters first:
#
#   Does doubling self-play search produce a better network PER GAME OF DATA?
#
# A side observation that sharpens it: in each arm's own history, sims32's
# vs_random plateaus near 0.75 from gen200 onward while sims64 climbs to 0.825.
# That is a saturated proxy and this project does not trust proxies over games
# -- but it points the same way, which is worth knowing before the matches run.
#
# If it does not, the per-hour question is already settled -- sims64 cost 1.53x
# the wall clock for the same 400 generations (8.58h against 5.61h), so it would
# have to win on data just to break even on time. Only if it wins here is the
# matched-wall-clock rerun worth renting hardware for.
#
# Head to head, not a proxy: this project has been wrong four times by trusting
# material, loss curves and tactical suites over actual games.
set -u
R=/Users/pontus/repos/chessbot
FC=${FC:-/Users/pontus/repos/fastchess/fastchess}
[ -x "$FC" ] || FC=$HOME/fastchess/fastchess
A=$R/runs/ab-salvage/ab-sims32
B=$R/runs/ab-salvage/ab-sims64
OUT=$R/runs/sims-equal-data
mkdir -p $OUT
L=$OUT/report.log
RAW=$OUT/raw.log

NODES=${NODES:-128}      # the match default; both sides at RootActions 32
# Concurrency = active searches, not engine processes: chess is turn-based, so
# while one engine thinks the other waits. P-cores, not logical cores -- a
# search scheduled onto an E-core drags the whole game it belongs to.
CONC=${CONC:-$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo 4)}

cd $R || exit 1
print "=== sims32 vs sims64 at equal data, started $(date) ===" | tee -a $L
print "both sides $NODES nodes, RootActions 32; only the self-play sims differ" | tee -a $L
print "concurrency $CONC (P-cores)" | tee -a $L
print "" | tee -a $L

duel () {  # generation, rounds, seed
  local gen=$1 rounds=$2 seed=$3
  local t0=$SECONDS
  $FC \
    -engine cmd=$R/.venv/bin/chess-uci args="--run $A" \
      name=sims32 option.Generation=$gen option.RootActions=32 nodes=$NODES \
    -engine cmd=$R/.venv/bin/chess-uci args="--run $B" \
      name=sims64 option.Generation=$gen option.RootActions=32 nodes=$NODES \
    -rounds $rounds -games 2 -concurrency $CONC -srand $seed \
    -openings file=$R/bench/openings-4ply.epd format=epd order=random \
    -log file=/dev/null >> $RAW 2>&1
  local e g
  e=$(grep -E "^Elo:" $RAW | tail -1)
  g=$(grep -E "^Games:" $RAW | tail -1)
  # Reported as sims32's margin, so POSITIVE means the CHEAPER arm won and the
  # extra search bought nothing.
  print "gen$gen  ::  $e  ::  $g  ::  $((SECONDS - t0))s" | tee -a $L
}

# A 16-game probe on 2026-10-19 put sims32 at 1W-14L-1D at gen400 -- about
# -394 Elo. Far too few games to report, but large enough to change what this
# run is for: the question is no longer WHETHER the extra search pays on data,
# it is WHEN the gap opens. So the budget is spread evenly across the curve
# rather than piled on the last point, and 150 games (+/-45 Elo) is ample
# resolution for a gap measured in hundreds.
#
# Ordered last-first so a short night still answers the primary question.
duel 400 ${R400:-75} 20261019
duel 200 ${R200:-75} 20261020
duel 300 ${R300:-75} 20261021
duel 100 ${R100:-75} 20261022
duel  50 ${R50:-75}  20261023

print "" | tee -a $L
print "timeouts: $(grep -c 'loses on time' $RAW 2>/dev/null || print 0)" | tee -a $L
print "=== COMPLETE $(date) ===" | tee -a $L
