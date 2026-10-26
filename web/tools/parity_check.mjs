// The gate: does the browser path compute what the trainer computes?
//
// Run against `web/dist` after a build. It loads the exact files the site will
// serve -- the same wasm, the same fp16 ONNX -- and holds them to the reference
// `tools/parity_ref.py` wrote straight out of the training code.
//
// Node rather than a real browser, deliberately. ONNX Runtime Web's wasm
// backend is the same code the page falls back to, so this tests a path that
// actually ships; a headless-Chrome WebGPU run would test the other path but
// needs a GPU in CI and fails for reasons that have nothing to do with the
// model. The WebGPU path is covered instead by the page's own load-time
// self-check, which compares the first evaluation against these same numbers
// and warns in the UI if the backend disagrees.
//
// Two kinds of claim, and the difference matters:
//
//   EXACT     FEN, legal move list and order, policy indices, and a hash of
//             the raw observation bytes. Same Rust, different target. A
//             tolerance here would hide the bug it is meant to catch.
//
//   TOLERANCE Network outputs, because ONNX Runtime is not PyTorch and fp16 is
//             not fp32. Judged on decisions -- the top legal move, and the
//             move a fixed-seed search ends up playing -- rather than on
//             floats, because a logit that moves by 1e-3 and changes nothing
//             costs nothing.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import * as ort from "onnxruntime-web";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.resolve(HERE, "..");
const DIST = process.env.DIST ?? path.join(WEB, "dist");

ort.env.wasm.wasmPaths = path.join(WEB, "node_modules", "onnxruntime-web", "dist") + path.sep;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error";

// Tolerances. The logit bound is generous because fp16 rounding on a logit is
// harmless; the bounds that actually gate the build are the agreement counts,
// which have to be perfect.
const LOGIT_TOL = 0.75;
const VALUE_TOL = 0.05;

let failures = 0;
const fail = (msg) => { failures++; console.log(`  FAIL  ${msg}`); };

function fnv1a(bytes) {
  let h = 0xcbf29ce484222325n;
  const M = 0x100000001b3n, MASK = 0xffffffffffffffffn;
  for (const b of bytes) h = ((h ^ BigInt(b)) * M) & MASK;
  return h.toString(16).padStart(16, "0");
}

// --- load the shipped artifacts --------------------------------------------

const ref = JSON.parse(readFileSync(path.join(WEB, "parity", "reference.json"), "utf8"));
const manifest = JSON.parse(readFileSync(path.join(DIST, "model", "manifest.json"), "utf8"));
const wasmBytes = readFileSync(path.join(DIST, "az_wasm.wasm"));
const modelBytes = readFileSync(path.join(DIST, "model", `${path.parse(manifest.checkpoint).name}.onnx`));

const { instance } = await WebAssembly.instantiate(wasmBytes, {});
const x = instance.exports;
const h = x.eng_new();
const bytes = () => new Uint8Array(x.memory.buffer);
const dec = new TextDecoder(), enc = new TextEncoder();
const str = (len) => {
  if (!len) return "";
  const p = x.eng_str_ptr(h);
  return dec.decode(bytes().subarray(p, p + len));
};
const withStr = (s, fn) => {
  const b = enc.encode(s), p = x.wasm_alloc(b.length);
  bytes().set(b, p);
  try { return fn(p, b.length); } finally { x.wasm_free(p, b.length); }
};
const setup = (moves) => {
  x.eng_reset(h);
  for (const m of moves) {
    if (withStr(m, (p, l) => x.eng_play_uci(h, p, l)) !== 0) throw new Error(`illegal in wasm: ${m}`);
  }
};

const session = await ort.InferenceSession.create(modelBytes, {
  executionProviders: ["wasm"],
  graphOptimizationLevel: "all",
});

console.log(`checkpoint   ${ref.checkpoint}`);
console.log(`model        ${manifest.precision}, ${(modelBytes.length / 1e6).toFixed(1)} MB`);
console.log(`core         ${(wasmBytes.length / 1e3).toFixed(0)} KB wasm`);
console.log(`cases        ${ref.cases.length} positions, ${ref.searches.length} searches\n`);

// --- 1. exact: encoding, move list, policy indices --------------------------

console.log("1. encoding and move indexing (exact)");
let worstLogit = 0, worstValue = 0, top1 = 0;
const obsLen = x.obs_floats();

for (const c of ref.cases) {
  setup(c.moves);
  const fen = str(x.eng_fen(h));
  if (fen !== c.fen) fail(`case ${c.id}: FEN ${fen} != ${c.fen}`);

  const legal = str(x.eng_legal_uci(h)).split(" ").filter(Boolean);
  if (legal.join(" ") !== c.legal.join(" ")) {
    fail(`case ${c.id}: legal moves differ (${legal.length} vs ${c.legal.length})`);
  }
  for (let i = 0; i < c.legal.length; i++) {
    const got = withStr(c.legal[i], (p, l) => x.eng_policy_index(h, p, l));
    if (got !== c.idx[i]) fail(`case ${c.id}: policy index for ${c.legal[i]} is ${got}, want ${c.idx[i]}`);
  }

  const ptr = x.eng_encode(h);
  const obs = new Float32Array(x.memory.buffer, ptr, obsLen).slice();
  const got = fnv1a(new Uint8Array(obs.buffer));
  if (got !== c.obs_hash) fail(`case ${c.id}: observation hash ${got} != ${c.obs_hash}`);
}
console.log(failures === 0 ? "   all exact matches\n" : "");

// --- 2. tolerance: the network -----------------------------------------------

console.log("2. network outputs (ONNX Runtime wasm vs torch)");
const before = failures;
for (const c of ref.cases) {
  setup(c.moves);
  const ptr = x.eng_encode(h);
  const obs = new Float32Array(x.memory.buffer, ptr, obsLen).slice();
  const out = await session.run({ obs: new ort.Tensor("float32", obs, [1, 119, 8, 8]) });

  const idx = c.legal.map((m) => withStr(m, (p, l) => x.eng_policy_index(h, p, l)));
  const got = idx.map((j) => out.policy.data[j]);
  for (let i = 0; i < got.length; i++) worstLogit = Math.max(worstLogit, Math.abs(got[i] - c.logits[i]));
  worstValue = Math.max(worstValue, Math.abs(out.value.data[0] - c.value));
  if (c.legal[got.indexOf(Math.max(...got))] === c.best_raw) top1++;
}
console.log(`   max |logit| diff   ${worstLogit.toFixed(5)}  (tolerance ${LOGIT_TOL})`);
console.log(`   max |value| diff   ${worstValue.toFixed(5)}  (tolerance ${VALUE_TOL})`);
console.log(`   top-1 legal move   ${top1}/${ref.cases.length} agree with torch`);
if (worstLogit > LOGIT_TOL) fail(`logit drift ${worstLogit} exceeds ${LOGIT_TOL}`);
if (worstValue > VALUE_TOL) fail(`value drift ${worstValue} exceeds ${VALUE_TOL}`);
if (failures === before) console.log("");

// --- 3. the whole thing: fixed-seed search ------------------------------------

console.log("3. fixed-seed search (wasm MCTS + ONNX vs native MCTS + ONNX)");
console.log("   the same seed and the same model must give the same move\n");
let agree = 0;
for (const sc of ref.searches) {
  setup(sc.moves);
  x.eng_search_start(h, sc.sims, sc.considered, BigInt(sc.seed));
  const logitPtr = x.wasm_alloc(x.policy_floats() * 4);
  while (x.eng_search_prepare(h) === 0) {
    const p = x.eng_search_encode(h);
    const obs = new Float32Array(x.memory.buffer, p, obsLen).slice();
    const out = await session.run({ obs: new ort.Tensor("float32", obs, [1, 119, 8, 8]) });
    new Float32Array(x.memory.buffer, logitPtr, x.policy_floats()).set(out.policy.data);
    x.eng_search_apply(h, logitPtr, out.value.data[0]);
  }
  x.wasm_free(logitPtr, x.policy_floats() * 4);

  const best = str(x.eng_search_best(h));
  const value = x.eng_search_value(h);
  const sims = x.eng_search_sims(h);
  const ok = best === sc.best && sims === sc.sims_done;
  if (ok) agree++;
  else fail(`search ${sc.id}: wasm plays ${best} (${sims} sims), native plays ${sc.best} (${sc.sims_done} sims)`);
  console.log(`   case ${String(sc.id).padStart(3)}  ${ok ? "ok" : "DIFFERS"}  ` +
              `${best}  value ${value >= 0 ? "+" : ""}${value.toFixed(4)} ` +
              `(native ${sc.value >= 0 ? "+" : ""}${sc.value.toFixed(4)})  ${sims} sims`);
}
console.log(`\n   ${agree}/${ref.searches.length} searches chose the same move`);

// --- verdict -----------------------------------------------------------------

console.log("\n" + "=".repeat(64));
if (failures === 0) {
  console.log("PARITY OK - the browser path matches the trainer. Safe to publish.");
  process.exit(0);
}
console.log(`PARITY FAILED - ${failures} problem(s). Do not publish.`);
process.exit(1);
