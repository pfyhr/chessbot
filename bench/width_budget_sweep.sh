#!/bin/bash
# Does the best root width track the node budget? -- rented GPU, Oct 2026.
#
# The 128-node row is already measured and is NOT repeated here:
#   ra4 -29 +/-36 | ra8 +41 +/-25 (pooled, 600 games) | ra16 +26 +/-40 | ra32 0
# and at 512 nodes ra8 came in at -57 +/-35. That is two points of a trend. This
# fills the rest of the grid so the trend is a shape rather than two cells.
#
# ra64 is deliberately absent: m = min(RootActions, legal moves) and the mean
# legal move count is 24.24, so ra32 already covers 96% of the move list and
# ra64 would be the identical search.
#
# Every match is gen695 against itself, booked, fixed nodes, both sides equal
# budget -- only RootActions differs. Cheapest budget first, and results append
# as they land, so a rental cut short still answers the question for the budgets
# it reached.
set -u
R=$HOME/chessbot
FC=$HOME/fastchess/fastchess
OUT=$R/runs/width-budget
mkdir -p "$OUT"
L=$OUT/report.log
RAW=$OUT/raw.log
CONC=${CONC:-12}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd "$R" || exit 1
say () { echo "$@" | tee -a "$L"; }

say "=== root width against node budget, started $(date) ==="
say "gen695 both sides; only RootActions differs; concurrency $CONC"
say "already known -- 128: ra4 -29, ra8 +41(pooled), ra16 +26 | 512: ra8 -57"
say ""

duel () {  # nodes, ra, rounds
    local nodes=$1 ra=$2 rounds=$3 t0=$SECONDS
    "$FC" \
        -engine cmd=$R/.venv/bin/chess-uci args="--run $R/runs/champ" \
            name=ra$ra option.Generation=695 option.RootActions=$ra nodes=$nodes \
        -engine cmd=$R/.venv/bin/chess-uci args="--run $R/runs/champ" \
            name=ra32 option.Generation=695 option.RootActions=32 nodes=$nodes \
        -rounds "$rounds" -games 2 -concurrency $CONC -srand $((nodes*1000+ra)) \
        -openings file=$R/bench/openings-4ply.epd format=epd order=random \
        -log file=/dev/null >>"$RAW" 2>&1
    local e g
    e=$(grep -E "^Elo:" "$RAW" | tail -1)
    g=$(grep -E "^Games:" "$RAW" | tail -1)
    say "$(printf '%5d nodes  ra%-3d over ra32' "$nodes" "$ra")  ::  $e  ::  $g  ::  $((SECONDS-t0))s"
}

say "--- 512 nodes (completing the row; ra8 already measured at -57 +/-35) ---"
duel  512  4 100
duel  512 16 100
say ""
say "--- 1024 nodes ---"
for ra in 4 8 16; do duel 1024 $ra 100; done
say ""
say "--- 2048 nodes (half the games; this rung is the expensive one) ---"
for ra in 4 8 16; do duel 2048 $ra 50; done
say ""
say "timeouts: $(grep -c 'loses on time' "$RAW" 2>/dev/null || echo 0)"
say "=== COMPLETE $(date) ==="
