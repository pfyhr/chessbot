#!/bin/zsh
#
# SCREENING ONLY. This answers "is this change worth adopting?" and nothing else.
#
# It runs a Sequential Probability Ratio Test, which stops the moment the
# evidence is decisive rather than after a fixed number of games. On the four
# architecture changes rejected in Sept-Oct 2026 that would have been dramatic:
#
#     attention trunk    -230 Elo   SPRT needs ~41 games    we played 300
#     PCR                -200 Elo   SPRT needs ~55 games    we played 200
#     narrow bottleneck  -133 Elo   SPRT needs ~99 games    we played 200
#     conv head          -125 Elo   SPRT needs ~107 games   we played 400
#
# What it CANNOT do, and must never be used for:
#
#   * It returns a verdict, not a magnitude. There is no "+91 +/-47" here. If a
#     number is going into bench/results/ or onto the status board, run a
#     fixed-length match instead -- this tool is not a measurement.
#   * It is not cheap for small or zero differences. A genuinely level pairing
#     needs ~535 games to reject, and a real 15 Elo gain needs ~1,600 to accept.
#     That is honest rather than a flaw: it tells you up front when a question is
#     expensive, instead of handing back a confident wrong answer from 200 games.
#
# Fixed nodes, not a clock, so a busy machine cannot distort the result.
#
#   bench/screen.sh runs/cap-big 385 runs/cap-small 574
#   bench/screen.sh runs/cap-big 385 runs/cap-small 574 25     # 25 Elo bar
#
set -u
R=/Users/pontus/repos/chessbot
FC=/Users/pontus/repos/fastchess/fastchess

if (( $# < 4 )); then
  print -u2 "usage: $0 <runA> <genA> <runB> <genB> [elo1=15] [max_rounds=2000]"
  print -u2 "       positive verdict means runA/genA is the better one"
  exit 2
fi
RA=$1 GA=$2 RB=$3 GB=$4
ELO1=${5:-15}
ROUNDS=${6:-2000}
OUT=$(mktemp -t screen)

print "SCREENING ONLY -- a verdict, not a measurement."
print "  A: $RA gen$GA"
print "  B: $RB gen$GB"
print "  H0: A is not more than 0 Elo better   H1: A is at least $ELO1 Elo better"
print "  alpha = beta = 0.05, fixed 128 nodes per move, up to $((ROUNDS*2)) games"
print ""

$FC \
  -engine cmd=$R/.venv/bin/chess-uci args="--run $R/$RA" name=A option.Generation=$GA nodes=128 \
  -engine cmd=$R/.venv/bin/chess-uci args="--run $R/$RB" name=B option.Generation=$GB nodes=128 \
  -sprt elo0=0 elo1=$ELO1 alpha=0.05 beta=0.05 model=normalized \
  -rounds $ROUNDS -games 2 -concurrency 4 \
  -openings file=$R/bench/openings-4ply.epd format=epd order=random \
  -log file=/dev/null 2>&1 | tee $OUT | grep -E "^(Elo|Games|LLR|SPRT)" | tail -4

played=$(grep -c "Finished game" $OUT)
llr=$(grep -oE "LLR: *-?[0-9.]+" $OUT | tail -1 | grep -oE -- "-?[0-9.]+")
print ""
print "games played: ${played:-0} of a possible $((ROUNDS*2))"
if grep -qi "H1 was accepted\|accepted H1" $OUT; then
  print "VERDICT: adopt -- A is at least $ELO1 Elo better"
  print "  next: run a fixed-length match if the magnitude needs recording"
elif grep -qi "H0 was accepted\|accepted H0" $OUT; then
  print "VERDICT: reject -- A is not $ELO1 Elo better"
else
  print "VERDICT: no decision within $((ROUNDS*2)) games (LLR ${llr:-?}, bounds +/-2.94)"
  print "  the difference is smaller than the bar you set; raise --elo1 or accept they are level"
fi
rm -f $OUT
