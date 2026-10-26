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
   * Fetch the weights and start a session.
   *
   * `onProgress(received, total)` is called while downloading; the model is
   * over 20 MB and a page that sits blank for that long looks broken.
   */
  static async load(url, { obsLen, policyLen, onProgress } = {}) {
    const bytes = await fetchWithProgress(url, onProgress);

    // Ask for WebGPU and accept wasm. ORT will itself fall back if a provider
    // fails to initialise, but asking for an unavailable provider throws on
    // some builds, so the capability is checked first.
    const wantGpu = typeof navigator !== "undefined" && "gpu" in navigator;
    const attempts = wantGpu ? [["webgpu"], ["wasm"]] : [["wasm"]];

    let lastErr;
    for (const providers of attempts) {
      try {
        const session = await ort.InferenceSession.create(bytes, {
          executionProviders: providers,
          graphOptimizationLevel: "all",
        });
        return new Net(session, { obsLen, policyLen, backend: providers[0] });
      } catch (e) {
        lastErr = e;
      }
    }
    throw new Error(`no ONNX backend would start: ${lastErr}`);
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
