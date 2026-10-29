#!/bin/zsh
# Published Elo for long-oct, plus the test that separates policy from search.
#
# SPRT said gen2600 beats gen1371 decisively, but on 52 games (+249 +/-128).
# SPRT stops as soon as a boundary is crossed, which is correct for a DECISION
# and far too wide to publish. Fixed length for the number that gets quoted.
#
# The second duel is the interesting one. long-oct's raw-policy probes fell over
# the run -- mate 0.023->0.012, defend 0.390->0.293 -- while the searched player
# apparently got much stronger. At 1 node there is no search, so the move is the
# policy's own. If the probes are right about the policy, gen1371 should do
# BETTER at 1 node than at 128 relative to gen2600.
set -u
# The repo root, from where this script lives, so a clone anywhere works.
R=$(cd "$(dirname "$0")/.." && pwd)
FC=${FC:-$(dirname "$R")/fastchess/fastchess}
OUT=$R/runs/long-oct-fixed
mkdir -p $OUT
L=$OUT/report.log
CONC=${CONC:-4}
ROUNDS=${ROUNDS:-150}   # 2 games per round

cd $R || exit 1
# fastchess autosaves its state to ./config.json; run it from a scratch dir so
# it stops dirtying the repo root. All paths below are absolute.
FCWORK=${TMPDIR:-/tmp}/fastchess-cwd
mkdir -p $FCWORK
print "=== long-oct fixed-length, started $(date) ===" | tee -a $L
print "$((ROUNDS*2)) games per duel, concurrency=$CONC (sharing with FEA)" | tee -a $L
print "positive Elo = long-oct/gen2600 won" | tee -a $L
print "" | tee -a $L

duel () {  # label, nodes, note
  local label=$1 nd=$2 note=$3
  local raw=$OUT/${label}.raw
  local t0=$SECONDS
  ( cd $FCWORK && $FC \
    -engine cmd=$R/.venv/bin/chess-uci args="--run $R/runs/long-oct" \
      name=gen2600 option.Generation=2600 option.RootActions=32 nodes=$nd \
    -engine cmd=$R/.venv/bin/chess-uci args="--run $R/runs/weekend" \
      name=gen1371 option.Generation=1371 option.RootActions=32 nodes=$nd \
    -rounds $ROUNDS -games 2 -concurrency $CONC -srand $((RANDOM)) \
    -openings file=$R/bench/openings-4ply.epd format=epd order=random \
    -log file=/dev/null > $raw 2>&1 )
  print "$label  ($nd nodes) -- $note" | tee -a $L
  print "    $(grep -E '^Games:' $raw | tail -1)" | tee -a $L
  print "    $(grep -E '^Elo:' $raw | tail -1)" | tee -a $L
  print "    took $((SECONDS-t0))s" | tee -a $L
  print "" | tee -a $L
}

duel searched 128 "the published number: both sides with search"
duel rawpolicy  1  "no search: is the raw policy better or worse than gen1371?"

print "=== COMPLETE $(date) ===" | tee -a $L
