#!/bin/sh
# Regenerate src/chess-pieces.woff2 -- six chess glyphs, nothing else.
#
# Run this only to update the font; the result is committed, so a normal build
# does not need fonttools and CI never runs this.
#
# Source: Noto Sans Symbols 2 (SIL OFL 1.1). It is the smallest freely
# redistributable face that covers U+265A..U+265F, and all six of its advances
# are exactly 1em, which is what makes the pieces metrically identical. The
# licence travels with the font in src/chess-pieces.OFL.txt -- keep them
# together, the OFL requires the notice to ship with the font.
#
# Why a bundled font at all: see the @font-face comment in src/style.css. The
# short version is that U+265F has an Emoji property and iOS Safari honours it.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
WEB=$(dirname "$HERE")
WORK=${WORK:-$(mktemp -d)}
BASE=https://raw.githubusercontent.com/google/fonts/main/ofl/notosanssymbols2

echo "work dir: $WORK"
python3 -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install -q fonttools brotli

curl -sSL -o "$WORK/NotoSansSymbols2-Regular.ttf" "$BASE/NotoSansSymbols2-Regular.ttf"
curl -sSL -o "$WORK/OFL.txt"                      "$BASE/OFL.txt"

# --layout-features='' and --no-hinting: nothing here shapes or hints, and they
# are most of the weight. --notdef-outline keeps a visible box if a seventh
# codepoint ever reaches this font by mistake, rather than a silent blank.
"$WORK/venv/bin/pyftsubset" "$WORK/NotoSansSymbols2-Regular.ttf" \
  --unicodes=U+265A-265F \
  --layout-features='' \
  --no-hinting \
  --desubroutinize \
  --name-IDs='' \
  --notdef-outline \
  --flavor=woff2 \
  --output-file="$WEB/src/chess-pieces.woff2"

cp "$WORK/OFL.txt" "$WEB/src/chess-pieces.OFL.txt"

ls -l "$WEB/src/chess-pieces.woff2"
echo "now re-run build.sh and tools/layout_check.mjs"
