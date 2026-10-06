#!/bin/bash
# What is search depth actually worth? Three studies. 6 Oct 2026, aida, gpu0.
#
# Why this exists: the leaf-depth histogram that started this was taken with
# `drivercost`, which feeds all-zero logits and values on purpose -- its job is
# to time the driver with the network excluded. Under a flat policy the interior
# rule argmax[pi'(a) - N(a)/(1+sum N)] degenerates to round-robin, the shallowest
# the search can be, so that histogram measured the benchmark, not the engine.
#
# 1. DEPTH   the histogram again, with gen695 actually driving the search,
#            across simulation budget and root width.
# 2. LADDER  what a doubling of search is worth in Elo: gen695 against itself at
#            adjacent node counts. Positive = the BIGGER budget won.
# 3. WIDTH   at a FIXED 128 nodes, trade root width for depth. RootActions 4/8/16
#            against the default 32. Positive = the NARROWER, deeper side won.
#            Costs nothing to adopt if it wins -- it is a flag, not a retrain.
#
# Both sides are the same network in every match, so these measure search alone.
# Results append to report.log as they land: a partial night is still useful.
set -u
R=$HOME/chessbot
FC=$HOME/fastchess/fastchess
OUT=$R/runs/night-depth
mkdir -p "$OUT"
L=$OUT/report.log
RAW=$OUT/raw.log
CONC=12                      # 24 engine procs, ~11 GB of the 24 GB card

export CUDA_VISIBLE_DEVICES=0
cd "$R" || exit 1

say () { echo "$@" | tee -a "$L"; }

say "=== search-depth study, started $(date) ==="
say "network: runs/champ/gen695.pt (w-long gen695, the 109h champion), both sides"
say ""

# --- 1. depth histogram, real network -------------------------------------
say "--- 1. leaf-depth histogram, gen695 driving the search ---"
.venv/bin/python bench/leaf_depth.py --run runs/champ \
    --sims 32,64,128,256,512 --considered 8,16,32 \
    --games 64 --concurrency 64 --out "$OUT/depth.json" 2>&1 | tee -a "$L"
say ""

# The counters cost time on every descent. Fixed-node matches are unaffected in
# strength but not in throughput, so rebuild clean before spending the night.
say "--- rebuilding the bridge without counters ---"
source "$HOME/.cargo/env"
if .venv/bin/maturin develop --release >>"$RAW" 2>&1; then say "    done"
else say "    REBUILD FAILED -- matches below still valid, just slower"; fi
say ""

# duel <label> <roundsq> <nodesA> <raA> <nodesB> <raB>
# Engine A is always the candidate, so a positive Elo means the candidate won.
duel () {
    local label=$1 rounds=$2 n1=$3 ra1=$4 n2=$5 ra2=$6
    local t0=$SECONDS
    "$FC" \
        -engine cmd=$R/.venv/bin/chess-uci args="--run $R/runs/champ" \
            name=A-n${n1}-ra${ra1} nodes=$n1 option.RootActions=$ra1 \
        -engine cmd=$R/.venv/bin/chess-uci args="--run $R/runs/champ" \
            name=B-n${n2}-ra${ra2} nodes=$n2 option.RootActions=$ra2 \
        -rounds "$rounds" -games 2 -concurrency $CONC \
        -openings file=$R/bench/openings-4ply.epd format=epd order=random \
        -log file=/dev/null >>"$RAW" 2>&1
    local e g
    e=$(grep -E "^Elo:" "$RAW" | tail -1)
    g=$(grep -E "^Games:" "$RAW" | tail -1)
    say "$label  ::  $e  ::  $g  ::  $((SECONDS - t0))s"
}

say "--- 2. what one doubling of search is worth (positive = more search won) ---"
# Rounds fall as the rungs get dearer: a game at 1024v512 costs sixteen times one
# at 64v32, and the cheap rungs are where the curve is most likely to bend.
# Sized off a calibration duel: 16 games of 128v32 took 92s at concurrency 8.
duel "LADDER   64 over   32" 150   64 32    32 32
duel "LADDER  128 over   64" 150  128 32    64 32
duel "LADDER  256 over  128" 120  256 32   128 32
duel "LADDER  512 over  256"  80  512 32   256 32
duel "LADDER 1024 over  512"  50 1024 32   512 32
say ""

say "--- 3. breadth vs depth at a fixed 128 nodes (positive = narrower/deeper won) ---"
duel "WIDTH    ra4 over ra32" 150 128  4   128 32
duel "WIDTH    ra8 over ra32" 150 128  8   128 32
duel "WIDTH   ra16 over ra32" 150 128 16   128 32
say ""

say "timeouts seen: $(grep -c 'loses on time' "$RAW" 2>/dev/null || echo 0)"
say "=== COMPLETE $(date) ==="
