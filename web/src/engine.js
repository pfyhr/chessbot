// The chess core, compiled to wasm, driven from JavaScript.
//
// `az_wasm.wasm` exports a flat C ABI and imports nothing, so there is no
// generated shim between here and the Rust -- this file *is* the binding. Two
// rules follow from that and are easy to get wrong:
//
//   1. Anything that allocates inside wasm can grow the memory, and growing it
//      detaches every existing TypedArray view. So views are made fresh at the
//      point of use, never cached across a call.
//   2. String results land in a scratch buffer inside the engine. Filling it
//      may move it, so `eng_str_ptr` is always read *after* the call that
//      wrote, never before.
//
// The search is the interesting part. `az-core`'s MCTS is a resumable state
// machine -- that is how the trainer pools leaf evaluations from hundreds of
// self-play games into one network batch -- which is exactly the shape an
// asynchronous evaluator needs. The browser runs the same loop with a batch of
// one.

const dec = new TextDecoder();
const enc = new TextEncoder();

/** Terminal status, as `eng_outcome` and `eng_search_proven` report it. */
const ONGOING = -1, DRAW = 0, WIN = 1, LOSS = 2;

export class Engine {
  #x; #h; #obsLen; #policyLen; #logitPtr;

  constructor(exports) {
    this.#x = exports;
    this.#h = exports.eng_new();
    this.#obsLen = exports.obs_floats();
    this.#policyLen = exports.policy_floats();
    // One permanent staging buffer for logits on their way back into wasm.
    this.#logitPtr = exports.wasm_alloc(this.#policyLen * 4);
  }

  static async load(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`engine: ${url} -> HTTP ${res.status}`);
    // instantiateStreaming needs the right Content-Type and some static hosts
    // get it wrong; the ArrayBuffer path always works and this file is small.
    const { instance } = await WebAssembly.instantiate(await res.arrayBuffer(), {});
    return new Engine(instance.exports);
  }

  get policyLen() { return this.#policyLen; }
  get obsLen() { return this.#obsLen; }

  // --- raw memory helpers -------------------------------------------------

  #bytes() { return new Uint8Array(this.#x.memory.buffer); }

  /** Copy a string into wasm memory, run `fn(ptr, len)`, free it. */
  #withStr(s, fn) {
    const b = enc.encode(s);
    const p = this.#x.wasm_alloc(b.length);
    this.#bytes().set(b, p);
    try { return fn(p, b.length); } finally { this.#x.wasm_free(p, b.length); }
  }

  /** Read the scratch buffer, given the length the filling call returned. */
  #str(len) {
    if (len === 0) return "";
    const p = this.#x.eng_str_ptr(this.#h);
    return dec.decode(this.#bytes().subarray(p, p + len));
  }

  // --- position -----------------------------------------------------------

  newGame() { this.#x.eng_reset(this.#h); }

  setFen(fen) { return this.#withStr(fen, (p, l) => this.#x.eng_set_fen(this.#h, p, l) === 0); }

  get fen() { return this.#str(this.#x.eng_fen(this.#h)); }
  get turn() { return this.#x.eng_turn(this.#h) === 0 ? "white" : "black"; }
  get ply() { return this.#x.eng_ply(this.#h); }
  get inCheck() { return this.#x.eng_in_check(this.#h) === 1; }

  legalMoves() {
    const s = this.#str(this.#x.eng_legal_uci(this.#h));
    return s ? s.split(" ") : [];
  }

  history() {
    const s = this.#str(this.#x.eng_history_uci(this.#h));
    return s ? s.split(" ") : [];
  }

  /** `{ from: [to, ...] }`, what a board widget needs to allow a drag. */
  dests() {
    const out = new Map();
    for (const m of this.legalMoves()) {
      const from = m.slice(0, 2), to = m.slice(2, 4);
      if (!out.has(from)) out.set(from, []);
      // A promotion is four moves sharing one destination; the board offers the
      // square once and asks which piece afterwards.
      if (!out.get(from).includes(to)) out.get(from).push(to);
    }
    return out;
  }

  /**
   * Terminal status, from the side to move's point of view.
   *
   * `az-core` treats fifty-move and threefold draws as terminal even though
   * legal moves remain, so "the game is over" is never inferred from an empty
   * move list here.
   */
  outcome() {
    switch (this.#x.eng_outcome(this.#h)) {
      case ONGOING: return null;
      case DRAW: return { kind: "draw", winner: null };
      case WIN: return { kind: "win", winner: this.turn };
      case LOSS: return { kind: "win", winner: this.turn === "white" ? "black" : "white" };
    }
  }

  play(uci) { return this.#withStr(uci, (p, l) => this.#x.eng_play_uci(this.#h, p, l) === 0); }

  undo() { return this.#x.eng_undo(this.#h) === 0; }

  policyIndex(uci) {
    const i = this.#withStr(uci, (p, l) => this.#x.eng_policy_index(this.#h, p, l));
    return i < 0 ? null : i;
  }

  /** The current position's observation planes, copied out. */
  encode() {
    const p = this.#x.eng_encode(this.#h);
    return new Float32Array(this.#x.memory.buffer, p, this.#obsLen).slice();
  }

  // --- search -------------------------------------------------------------

  /**
   * Run a full Gumbel MCTS from the current position.
   *
   * `evaluate(obs)` returns `{ policy, value }` for one position: `policy` is
   * `policyLen` logits and `value` is P(win) - P(loss) from that position's
   * mover's point of view.
   *
   * `seed` fixes the root Gumbel perturbation, so the same seed and the same
   * network give the same move -- which is what makes the parity test possible
   * at all.
   */
  async search(evaluate, { sims = 128, considered = 32, seed = 1, onProgress } = {}) {
    this.#x.eng_search_start(this.#h, sims, considered, BigInt(seed));

    let done = 0;
    while (this.#x.eng_search_prepare(this.#h) === 0) {
      const ptr = this.#x.eng_search_encode(this.#h);
      // Copied, not viewed: the evaluator is async, and anything that allocates
      // in wasm while it is awaited would detach a view handed out here.
      const obs = new Float32Array(this.#x.memory.buffer, ptr, this.#obsLen).slice();
      const { policy, value } = await evaluate(obs);
      new Float32Array(this.#x.memory.buffer, this.#logitPtr, this.#policyLen).set(policy);
      this.#x.eng_search_apply(this.#h, this.#logitPtr, value);
      if (onProgress && (++done & 7) === 0) onProgress(done, sims);
    }

    const best = this.#str(this.#x.eng_search_best(this.#h));
    const detail = JSON.parse(this.#str(this.#x.eng_search_detail(this.#h, 8)));
    const proven = this.#x.eng_search_proven(this.#h);
    return {
      move: best,
      value: this.#x.eng_search_value(this.#h),
      sims: this.#x.eng_search_sims(this.#h),
      proven: proven === ONGOING ? null : proven === DRAW ? "draw" : proven === WIN ? "win" : "loss",
      detail,
    };
  }

  /**
   * The raw network's choice, with no search at all.
   *
   * Worth having as its own mode: it is how the policy head is measured in the
   * repo's own match scripts, and at one node this network is only a little
   * weaker than it is at 128.
   */
  async rawBest(evaluate) {
    const { policy, value } = await evaluate(this.encode());
    let best = null, top = -Infinity;
    for (const m of this.legalMoves()) {
      const v = policy[this.policyIndex(m)];
      if (v > top) { top = v; best = m; }
    }
    return { move: best, value, sims: 0, proven: null, detail: [] };
  }
}
