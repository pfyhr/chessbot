// Does the board render as a board?
//
// Added after a shipped bug that no existing gate could have caught: the board
// had `aspect-ratio: 1`, so the *container* was square, but its eight rows were
// created implicitly and sized `auto` -- to the height of a piece glyph rather
// than to an eighth of the board. The squares came out wider than they were
// tall, on a page whose search, encoding and network were all verified exact.
//
// Geometry needs a layout engine, so this is the one gate that runs a real
// browser. jsdom and friends parse CSS without ever computing a box, and would
// have passed the broken build happily.
//
// It loads the built site over HTTP exactly as a visitor does -- which also
// means it covers module loading, the ONNX session starting, and the page's own
// self-check passing, none of which the Node gates exercise.

import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { chromium, webkit } from "playwright";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DIST = process.env.DIST ?? path.join(HERE, "..", "dist");

const TYPES = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".css": "text/css", ".json": "application/json", ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
};

// Both engines, because the bug that prompted this was invisible in one of
// them. With the fix reverted, headless Chromium still produced square squares
// and was only caught by the "8 squares must fill the board" check, off by 2px;
// WebKit is where empty squares visibly collapsed, because it sizes an implicit
// `auto` row from its content and Chromium stretches it. Testing one engine
// would have shipped this twice.
const ENGINES = [["chromium", chromium], ["webkit", webkit]];

// The mate from a real game on the live page, which ended without the page
// ever saying so. Not a synthetic position: the regression was that the result
// was computed correctly and then written into a hidden element.
const MATES = [
  { name: "Qxh5#", plies: 19,
    uci: "e2e4,h7h6,g2g3,a7a6,g1f3,b7b5,d2d4,h6h5,c1g5,c7c6,b1c3,g7g6," +
         "f3e5,f7f6,e5g6,c8b7,g6h8,f6g5,d1h5" },
  // The second game from the live page. It ended with no result shown AND the
  // evaluation panel still listing the previous search's candidates, which were
  // not legal in the position on screen.
  { name: "Bxf7#", plies: 13,
    uci: "e2e4,h7h6,g1f3,h6h5,d2d4,g8f6,f1b5,f6e4,f3e5,g7g5,b5c4,f7f5,c4f7" },
];

// Phone, small tablet, and the two-column desktop layout -- the last is where
// the shipped bug was worst, because the board is one column of a grid there
// and a viewport-relative glyph size has nothing to do with the board's width.
const VIEWPORTS = [
  { name: "phone",   width: 390,  height: 844 },
  { name: "tablet",  width: 820,  height: 1180 },
  { name: "desktop", width: 1440, height: 900 },
];

// One CSS pixel of slack: fractional layout is normal, a non-square board is not.
const TOL = 1.0;

let failures = 0;
let engineName = "";
const fail = (m) => { failures++; console.log(`  FAIL  [${engineName}] ${m}`); };
const near = (a, b) => Math.abs(a - b) <= TOL;

const server = createServer(async (req, res) => {
  const rel = decodeURIComponent(req.url.split("?")[0]).replace(/^\/+/, "") || "index.html";
  const file = path.join(DIST, rel);
  if (!file.startsWith(DIST)) { res.writeHead(403).end(); return; }
  try {
    const body = await readFile(file);
    res.writeHead(200, { "content-type": TYPES[path.extname(file)] ?? "application/octet-stream" });
    res.end(body);
  } catch { res.writeHead(404).end("not found"); }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${server.address().port}/`;

console.log(`layout check against ${DIST}\n`);

for (const [name, engine] of ENGINES) {
engineName = name;
const browser = await engine.launch();
console.log(`### ${name}\n`);

for (const vp of VIEWPORTS) {
  const page = await browser.newPage({ viewport: { width: vp.width, height: vp.height } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });

  await page.goto(base, { waitUntil: "domcontentloaded" });
  // The app only unhides once the network has downloaded, the session has
  // started and the self-check has run, so waiting on it covers all three.
  await page.waitForSelector("#app:not(.hidden)", { timeout: 120_000 });

  const m = await page.evaluate(() => {
    const board = document.getElementById("board");
    const b = board.getBoundingClientRect();
    // `clientWidth/Height` is the content box. The bounding rect includes the
    // 1px border on each side, so squares tile 710px of a 712px rect -- a
    // difference that is correct and must not be read as a layout bug.
    const inner = { w: board.clientWidth, h: board.clientHeight };
    const sq = [...board.querySelectorAll(".sq")].map((e) => {
      const r = e.getBoundingClientRect();
      return { sq: e.dataset.square, w: r.width, h: r.height };
    });
    const warn = document.getElementById("warn");
    return {
      board: { w: b.width, h: b.height },
      inner,
      squares: sq,
      count: sq.length,
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      pieces: board.querySelectorAll(".piece").length,
      warned: warn && !warn.classList.contains("hidden") ? warn.textContent : null,
      backend: document.getElementById("netinfo").textContent,
    };
  });

  const widths = m.squares.map((s) => s.w), heights = m.squares.map((s) => s.h);
  const wMin = Math.min(...widths), wMax = Math.max(...widths);
  const hMin = Math.min(...heights), hMax = Math.max(...heights);

  console.log(`${vp.name} (${vp.width}x${vp.height})`);
  console.log(`  board    ${m.board.w.toFixed(1)} x ${m.board.h.toFixed(1)}  (content ${m.inner.w.toFixed(1)} x ${m.inner.h.toFixed(1)})`);
  console.log(`  square   w ${wMin.toFixed(2)}-${wMax.toFixed(2)}  h ${hMin.toFixed(2)}-${hMax.toFixed(2)}`);
  console.log(`  pieces   ${m.pieces}   overflow ${m.overflow}px`);

  if (m.count !== 64) fail(`${vp.name}: ${m.count} squares, expected 64`);
  if (!near(m.board.w, m.board.h)) {
    fail(`${vp.name}: board is ${m.board.w.toFixed(1)}x${m.board.h.toFixed(1)}, not square`);
  }
  // Every square square, and all of them the same size as each other.
  for (const s of m.squares) {
    if (!near(s.w, s.h)) {
      fail(`${vp.name}: square ${s.sq} is ${s.w.toFixed(2)}x${s.h.toFixed(2)}, not square`);
      break;  // one example is enough; 64 identical failures is just noise
    }
  }
  if (!near(wMin, wMax)) fail(`${vp.name}: square widths vary, ${wMin.toFixed(2)}-${wMax.toFixed(2)}`);
  if (!near(hMin, hMax)) fail(`${vp.name}: square heights vary, ${hMin.toFixed(2)}-${hMax.toFixed(2)}`);
  // 8 squares must tile the board exactly; a rounding drift would show here.
  if (!near(wMin * 8, m.inner.w)) {
    fail(`${vp.name}: 8 squares of ${wMin.toFixed(2)} do not fill ${m.inner.w.toFixed(1)}px of board`);
  }
  if (!near(hMin * 8, m.inner.h)) {
    fail(`${vp.name}: 8 rows of ${hMin.toFixed(2)} do not fill ${m.inner.h.toFixed(1)}px of board`);
  }
  if (m.pieces !== 32) fail(`${vp.name}: ${m.pieces} pieces on a starting board, expected 32`);
  if (m.overflow > 0) fail(`${vp.name}: page scrolls horizontally by ${m.overflow}px`);
  if (m.warned) fail(`${vp.name}: page raised a warning: ${m.warned}`);
  if (errors.length) fail(`${vp.name}: console errors: ${errors.slice(0, 3).join(" | ")}`);
  if (vp.name === "desktop") console.log(`  net      ${m.backend}`);
  console.log("");
  await page.close();
}

// --- the game actually ends ------------------------------------------------
//
// Loaded through `?moves=`, which is the page's own feature rather than a
// test-only hook, so this exercises the path a visitor takes.
for (const mate of MATES) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(`${base}?moves=${mate.uci}&side=white`, { waitUntil: "domcontentloaded" });
  await page.waitForSelector("#app:not(.hidden)", { timeout: 120_000 });

  const plies = () => page.evaluate(() =>
    document.getElementById("moves").textContent.trim().split(/\s+/)
      .filter((t) => !/^\d+\.$/.test(t)).length);

  const r = await page.evaluate(() => ({
    status: document.getElementById("status").textContent.trim(),
    movable: document.querySelectorAll("#board .piece.movable").length,
    // Candidate moves left over from an earlier position read as the engine's
    // current opinion, and are often not even legal.
    lines: document.querySelectorAll("#lines li").length,
    warned: (() => { const w = document.getElementById("warn");
                     return w && !w.classList.contains("hidden") ? w.textContent.trim() : null; })(),
    backend: document.getElementById("netinfo").textContent,
  }));
  const n = await plies();

  console.log(`### terminal state: ${mate.name}`);
  console.log(`  status   "${r.status}"`);
  console.log(`  plies    ${n}   movable ${r.movable}   stale eval rows ${r.lines}`);

  if (!r.status.includes("White wins by checkmate")) {
    fail(`${mate.name}: game over not reported, got "${r.status}"`);
  }
  if (n !== mate.plies) fail(`${mate.name}: ${n} plies replayed, expected ${mate.plies}`);
  if (r.movable !== 0) fail(`${mate.name}: ${r.movable} pieces still movable after mate`);
  if (r.lines !== 0) fail(`${mate.name}: evaluation panel shows ${r.lines} stale candidate moves`);
  if (r.warned) fail(`${mate.name}: page raised a warning: ${r.warned}`);

  // And the engine must not answer a finished game.
  await page.waitForTimeout(1500);
  if (await plies() !== mate.plies) fail(`${mate.name}: the engine moved after mate`);
  if (errors.length) fail(`${mate.name}: console errors: ${errors.slice(0, 3).join(" | ")}`);
  console.log(`  backend  ${r.backend}\n`);
  await page.close();
}

await browser.close();
}

server.close();

console.log("=".repeat(64));
console.log(failures === 0 ? "LAYOUT OK - the board renders as a board."
                           : `LAYOUT FAILED - ${failures} problem(s).`);
process.exit(failures === 0 ? 0 : 1);
