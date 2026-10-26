// Exercise the module the page actually imports.
//
// `parity_check.mjs` talks to the wasm ABI directly, on purpose: it is testing
// the Rust, and going through the glue would let a glue bug mask a core bug.
// That leaves `src/engine.js` itself untested -- the memory-view handling, the
// async search driver, the dests map -- which is the code most likely to break
// on a refactor and least likely to fail loudly. So this plays a short game
// through it and insists the result is legal chess.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Engine } from "../src/engine.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.resolve(HERE, "..");
const DIST = process.env.DIST ?? path.join(WEB, "dist");
const PLIES = Number(process.env.PLIES ?? 12);
const SIMS = Number(process.env.SIMS ?? 16);

// NOTE: ONNX Runtime is imported from `dist/vendor`, not from `node_modules`.
// That is deliberate. Vendoring ORT means copying several files that depend on
// each other -- the bundle dynamically imports the wasm glue, which fetches the
// binary -- and missing one produces "no available backend found" at runtime
// while every other part of the page works. Testing the npm package instead of
// the copied files would not have noticed.
const ort = await import(pathToFileURL(
  path.join(DIST, "vendor", "ort", "ort.bundle.min.mjs")).href);
ort.env.wasm.wasmPaths = pathToFileURL(path.join(DIST, "vendor", "ort") + path.sep).href;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error";

const manifest = JSON.parse(readFileSync(path.join(DIST, "model", "manifest.json"), "utf8"));
const { instance } = await WebAssembly.instantiate(
  readFileSync(path.join(DIST, "az_wasm.wasm")), {});
const eng = new Engine(instance.exports);

const session = await ort.InferenceSession.create(
  readFileSync(path.join(DIST, "model", `${path.parse(manifest.checkpoint).name}.onnx`)),
  { executionProviders: ["wasm"], graphOptimizationLevel: "all" });

const evaluate = async (obs) => {
  const out = await session.run({ obs: new ort.Tensor("float32", obs, [1, 119, 8, 8]) });
  return { policy: out.policy.data, value: out.value.data[0] };
};

let problems = 0;
const check = (cond, msg) => { if (!cond) { problems++; console.log(`  FAIL  ${msg}`); } };

console.log(`engine.js smoke test: ${PLIES} plies at ${SIMS} sims\n`);
eng.newGame();
check(eng.ply === 0, "a new game starts at ply 0");
check(eng.turn === "white", "White moves first");
check(eng.legalMoves().length === 20, `20 legal moves at the start, got ${eng.legalMoves().length}`);

// The dests map is what the board allows a drag to, so it has to describe
// exactly the moves the core calls legal -- no more, and no fewer.
const dests = eng.dests();
const flat = new Set();
for (const [from, tos] of dests) for (const to of tos) flat.add(from + to);
for (const m of eng.legalMoves()) {
  check(flat.has(m.slice(0, 4)), `dests is missing ${m}`);
}

const started = Date.now();
let evals = 0;
const counting = async (obs) => { evals++; return evaluate(obs); };

for (let i = 0; i < PLIES; i++) {
  if (eng.outcome()) break;
  const before = eng.fen;
  const res = await eng.search(counting, { sims: SIMS, considered: 16, seed: i + 1 });
  check(eng.legalMoves().includes(res.move), `search returned an illegal move: ${res.move}`);
  check(Number.isFinite(res.value), `value is not finite: ${res.value}`);
  check(res.detail.length > 0, "root detail came back empty");
  const sum = res.detail.reduce((a, r) => a + r.p, 0);
  check(sum > 0 && sum <= 1.0001, `improved policy over the top moves sums to ${sum}`);
  check(eng.play(res.move), `could not play its own move ${res.move}`);
  check(eng.fen !== before, "the position did not change");
  process.stdout.write(`  ${i + 1}. ${res.move} (${res.value >= 0 ? "+" : ""}${res.value.toFixed(3)})\n`);
}

// Take-back must land back where it was, history included.
const fenBefore = eng.fen, plyBefore = eng.ply;
check(eng.undo(), "undo failed");
check(eng.ply === plyBefore - 1, "undo did not drop a ply");
const back = eng.history();
check(back.length === plyBefore - 1, "history did not shrink with the undo");
check(eng.fen !== fenBefore, "undo left the position unchanged");

// And the raw-policy path, which skips search entirely.
const raw = await eng.rawBest(evaluate);
check(eng.legalMoves().includes(raw.move), `raw policy returned an illegal move: ${raw.move}`);
check(raw.sims === 0, "raw policy should report zero simulations");

const secs = (Date.now() - started) / 1000;
console.log(`\n  ${evals} evaluations in ${secs.toFixed(1)}s (${(evals / secs).toFixed(1)}/s, wasm CPU)`);
console.log(problems === 0 ? "\nSMOKE OK" : `\nSMOKE FAILED - ${problems} problem(s)`);
process.exit(problems === 0 ? 0 : 1);
