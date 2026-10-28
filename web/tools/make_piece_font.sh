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
  --output-file="$WORK/subset.ttf"

# Centre the glyphs in their line box, then compress.
#
# Noto's vertical metrics are built for a line of text: ascent 1069, descent
# -630, so the box centre sits at (1069-630)/2 = +219 above the baseline. The
# six chess glyphs occupy -49..767, centred at +359. The ink therefore sits
# 140 units -- 0.14em -- above the middle of its own box, which on a phone is
# about 4.6px on a 45px square, and every piece rode high.
#
# The fix belongs in the font, not in CSS. `ascent-override` only reached
# Safari 17, so on an older iPhone it would silently do nothing. Metrics baked
# into the file work on any engine that can load it at all.
#
# Values are DERIVED from the glyph outlines, never typed in: make the content
# box one em and put its centre on the ink's centre.
"$WORK/venv/bin/python" - "$WORK/subset.ttf" "$WEB/src/chess-pieces.woff2" <<'PYEOF'
import sys
from fontTools.ttLib import TTFont
from fontTools.pens.boundsPen import BoundsPen

src, dst = sys.argv[1], sys.argv[2]
f = TTFont(src)
upem = f["head"].unitsPerEm
gs, cmap = f.getGlyphSet(), f.getBestCmap()

lo, hi = None, None
for cp in range(0x265A, 0x2660):
    g = cmap.get(cp)
    if not g:
        continue
    bp = BoundsPen(gs)
    gs[g].draw(bp)
    if bp.bounds:
        _, y0, _, y1 = bp.bounds
        lo = y0 if lo is None else min(lo, y0)
        hi = y1 if hi is None else max(hi, y1)
if lo is None:
    raise SystemExit("no chess glyph outlines found")

centre = (lo + hi) / 2
ascent = round(upem / 2 + centre)
descent = round(upem / 2 - centre)          # positive magnitude
# The box must still contain the ink; widen symmetrically if it would not.
grow = max(0, round(hi) - ascent, round(-lo) - descent)
ascent += grow
descent += grow

f["hhea"].ascent, f["hhea"].descent, f["hhea"].lineGap = ascent, -descent, 0
os2 = f["OS/2"]
os2.sTypoAscender, os2.sTypoDescender, os2.sTypoLineGap = ascent, -descent, 0
os2.usWinAscent, os2.usWinDescent = ascent, descent
os2.fsSelection |= 1 << 7                   # USE_TYPO_METRICS, so all engines agree
f.flavor = "woff2"
f.save(dst)

print(f"ink {lo:.0f}..{hi:.0f} centre {centre:.0f}  ->  ascent {ascent} descent -{descent} "
      f"(box centre {(ascent - descent) / 2:.0f})")
PYEOF

cp "$WORK/OFL.txt" "$WEB/src/chess-pieces.OFL.txt"

ls -l "$WEB/src/chess-pieces.woff2"
echo "now re-run build.sh and tools/layout_check.mjs"
