// The network, as ONNX Runtime Web.
//
// WebGPU where it exists, the SIMD wasm kernels where it does not. The page is
// usable either way; the difference is roughly whether a 128-simulation move
// takes under a second or a few seconds, and the readout says which one you
// got rather than leaving it a mystery.
//
// The model is fp16. That was measured, not assumed: against the fp32 export,
// over 127 positions, fp16 moves the top policy move in two of them -- both
// ties the network itself scores equal to four decimal places -- and changes
// no move at all in 12 fixed-seed searches. See `bench/results/`.

import * as ort from "../vendor/ort/ort.bundle.min.mjs";

ort.env.wasm.wasmPaths = new URL("../vendor/ort/", import.meta.url).href;
// GitHub Pages cannot set COOP/COEP, so there is no SharedArrayBuffer and wasm
// threads are unavailable no matter what we ask for. Saying 1 up front stops
// ORT probing for workers it will never get. WebGPU is unaffected -- it is the
// path that matters for speed anyway.
ort.env.wasm.numThreads = 1;
// Logging off: ORT is chatty on first run and the page has its own status line.
ort.env.logLevel = "error";

export class Net {
  #session; #obsLen; #policyLen;

  constructor(session, { obsLen, policyLen, backend }) {
    this.#session = session;
    this.#obsLen = obsLen;
    this.#policyLen = policyLen;
    this.backend = backend;
  }

  /**
   * Fetch the weights and start a session on a backend that passes `verify`.
   *
   * `verify(net)` returns the worst absolute deviation from the reference
   * vectors; a backend is only used if that is within `tolerance`. This is not
   * belt-and-braces. Safari's WebGPU fp16 path was measured at 0.72 worst-case
   * on a logit and 0.14 on the value head against ONNX Runtime on CPU -- it
   * changed the top policy move in 5 of 127 positions and the search's chosen
   * move in 1 of 12, while the Node and Chromium gates passed. A backend that
   * is subtly wrong is worse than one that fails, because nothing reports it.
   *
   * CPU first, which is not the obvious order and is what the measurements
   * say: this search evaluates one leaf at a time, so a GPU pays full dispatch
   * overhead on a batch of one. Safari's WebGPU managed 68 evaluations/second
   * where the wasm kernels do 205 on the same machine. The GPU is both less
   * accurate and three times slower here, so it is opt-in via `?backend=webgpu`
   * rather than preferred.
   */
  static async load(url, { obsLen, policyLen, onProgress, verify, tolerance = Infinity,
                           prefer = null } = {}) {
    const bytes = await fetchWithProgress(url, onProgress);

    const hasGpu = typeof navigator !== "undefined" && "gpu" in navigator;
    const order = prefer === "webgpu" && hasGpu ? ["webgpu", "wasm"] : ["wasm"];

    const tried = [];
    let best = null;
    for (const backend of order) {
      let session;
      try {
        session = await ort.InferenceSession.create(bytes, {
          executionProviders: [backend],
          graphOptimizationLevel: "all",
        });
      } catch (e) {
        tried.push(`${backend}: would not start (${e.message})`);
        continue;
      }
      const net = new Net(session, { obsLen, policyLen, backend });
      if (!verify) return net;

      const worst = await verify(net);
      net.deviation = worst;
      if (worst <= tolerance) {
        net.rejected = tried;
        return net;
      }
      tried.push(`${backend}: self-check off by ${worst.toExponential(1)}`);
      best ??= net;
      await session.release?.();
    }

    if (!best) throw new Error(`no ONNX backend would start: ${tried.join("; ")}`);
    // Nothing passed. Hand back the least-bad one and let the page decide
    // whether to play with it -- but it has to be told, not guessed at.
    best.rejected = tried;
    best.unverified = true;
    return best;
  }

  /**
   * One position in, policy logits and a scalar value out.
   *
   * The value reduction -- P(win) - P(loss) over the WDL head -- is folded into
   * the exported graph rather than done here, so there is one fewer place for
   * the browser and the trainer to disagree.
   */
  async evaluate(obs) {
    const input = new ort.Tensor("float32", obs, [1, 119, 8, 8]);
    const out = await this.#session.run({ obs: input });
    return {
      policy: out.policy.data,
      value: out.value.data[0],
      wdl: Array.from(out.wdl.data),
    };
  }
}

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`model: ${url} -> HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !onProgress) return new Uint8Array(await res.arrayBuffer());

  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received, total);
  }
  const buf = new Uint8Array(received);
  let at = 0;
  for (const c of chunks) { buf.set(c, at); at += c.length; }
  return buf;
}
