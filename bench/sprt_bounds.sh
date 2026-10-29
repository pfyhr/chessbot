#!/bin/zsh
# Pick SPRT bounds that terminate at this project's scale -- M2, Oct 2026.
#
# The [0,10] validation gave every verdict right and was useless in practice:
# 244 games to confirm a +197 Elo effect, and 800 games on a true null without
# getting a seventh of the way to a boundary. Those bounds are built for engine
# developers chasing 2-5 Elo. Our effects are tens to hundreds.
#
# Two tests that matter more than a synthetic one:
#
#   REGRET  ra8 against ra32 at 128 nodes. Pooled over 600 games this is
#           +41 +/-25 -- real, but it was first reported as +58 +/-34 and
#           nearly adopted as a default. With a 50-Elo bar, SPRT should
#           REFUSE it. That is the exact mistake we want the method to prevent.
#   NULL    identical engines. The binding practical question is whether H0 is
#           reachable at all in a tolerable number of games.
set -u
# The repo root, from where this script lives, so a clone anywhere works.
R=$(cd "$(dirname "$0")/.." && pwd)
FC=${FC:-$(dirname "$R")/fastchess/fastchess}
OUT=$R/runs/sprt-bounds
mkdir -p $OUT
L=$OUT/report.log
CONC=${CONC:-$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo 8)}
E1=${E1:-50}       # "worth changing the default for" threshold
CAP=${CAP:-700}    # rounds

cd $R || exit 1
# fastchess autosaves its state to ./config.json; run it from a scratch dir so
# it stops dirtying the repo root. All paths below are absolute.
FCWORK=${TMPDIR:-/tmp}/fastchess-cwd
mkdir -p $FCWORK
print "=== SPRT bounds [0,$E1], started $(date) ===" | tee -a $L
print "alpha=0.05 beta=0.05 model=logistic cap=$((CAP*2)) games conc=$CONC" | tee -a $L
print "" | tee -a $L

duel () {  # label, ra_a, ra_b, nodes, seed, expect
  local label=$1 ra=$2 rb=$3 nd=$4 seed=$5 expect=$6
  local raw=$OUT/$label.raw t0=$SECONDS
  ( cd $FCWORK && $FC \
    -engine cmd=$R/.venv/bin/chess-uci args="--run $R/runs/w-long" \
      name=cand option.Generation=695 option.RootActions=$ra nodes=$nd \
    -engine cmd=$R/.venv/bin/chess-uci args="--run $R/runs/w-long" \
      name=base option.Generation=695 option.RootActions=$rb nodes=$nd \
    -sprt elo0=0 elo1=$E1 alpha=0.05 beta=0.05 model=logistic \
    -rounds $CAP -games 2 -concurrency $CONC -srand $seed \
    -openings file=$R/bench/openings-4ply.epd format=epd order=random \
    -log file=/dev/null > $raw 2>&1 )
  {
    print "$label  (ra$ra vs ra$rb at $nd nodes; expect: $expect)"
    print "    $(grep -E '^Games:' $raw | tail -1)"
    print "    $(grep -E '^Elo:' $raw | tail -1)"
    print "    $(grep -iE 'LLR' $raw | tail -1)"
    print "    verdict: $(grep -iE 'H1 was accepted|H0 was accepted' $raw | tail -1 || print 'no boundary crossed within cap')"
    print "    took $((SECONDS-t0))s"
    print ""
  } | tee -a $L
}

duel "REGRET" 8  32 128 20261110 "H0 — +41 is real but under the 50-Elo bar"
duel "NULL"   32 32 128 20261111 "H0 — and the question is how many games"

print "=== COMPLETE $(date) ===" | tee -a $L
