"""The browser front end for `chess-serve`, as a single self-contained page."""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>chessbot</title>
<style>
  :root {
    color-scheme: light;
    --ground:#eef1f5; --panel:#fbfcfd; --ink:#0d1620; --muted:#5c6a7c;
    --line:#d8e0ea; --accent:#2a78d6;
    --light-sq:#e6ccb0; --dark-sq:#b3855c; --sel:#8fbf6a; --hint:#7fae5e;
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      color-scheme: dark;
      --ground:#0e1218; --panel:#161c24; --ink:#e8eef6; --muted:#8593a5;
      --line:#242e3a; --accent:#3987e5;
      --light-sq:#9c8368; --dark-sq:#6b5540; --sel:#6d9c52; --hint:#5e8b46;
    }
  }
  * { box-sizing:border-box; }
  body {
    margin:0; padding:20px; background:var(--ground); color:var(--ink);
    font-family:var(--sans); display:flex; gap:20px; flex-wrap:wrap;
    justify-content:center; align-items:flex-start;
  }
  h1 { font-family:var(--mono); font-size:.95rem; letter-spacing:.06em;
       text-transform:uppercase; color:var(--muted); margin:0 0 12px; }

  #board { display:grid; grid-template-columns:repeat(8, min(11vw, 68px));
           grid-template-rows:repeat(8, min(11vw, 68px));
           border:1px solid var(--line); border-radius:6px; overflow:hidden; }
  .sq { display:flex; align-items:center; justify-content:center;
        font-size:min(8vw,46px); line-height:1; cursor:pointer; position:relative;
        user-select:none; }
  .sq.light { background:var(--light-sq); } .sq.dark { background:var(--dark-sq); }
  .sq.sel { background:var(--sel) !important; }
  .sq.last::after { content:""; position:absolute; inset:0;
                    box-shadow: inset 0 0 0 3px var(--accent); opacity:.55; }
  .sq .dot { position:absolute; width:26%; height:26%; border-radius:50%;
             background:var(--hint); opacity:.85; pointer-events:none; }
  .sq.occupied .dot { width:88%; height:88%; border-radius:50%; background:none;
                      box-shadow: inset 0 0 0 4px var(--hint); opacity:.8; }
  .wp { color:#fff; text-shadow:0 1px 2px rgba(0,0,0,.45); }
  .bp { color:#151515; text-shadow:0 1px 1px rgba(255,255,255,.25); }

  .side { width:270px; display:flex; flex-direction:column; gap:12px; }
  .card { background:var(--panel); border:1px solid var(--line);
          border-radius:8px; padding:12px; }
  label { font-family:var(--mono); font-size:.68rem; letter-spacing:.1em;
          text-transform:uppercase; color:var(--muted); display:block; margin-bottom:5px; }
  select, input[type=range] { width:100%; }
  button { font-family:var(--mono); font-size:.72rem; letter-spacing:.05em;
           text-transform:uppercase; padding:7px 10px; border-radius:5px;
           border:1px solid var(--line); background:var(--panel); color:var(--ink);
           cursor:pointer; }
  button:hover { border-color:var(--accent); color:var(--accent); }
  .row { display:flex; gap:8px; } .row > * { flex:1; }

  #evalwrap { height:12px; background:var(--line); border-radius:6px; overflow:hidden; }
  #evalbar { height:100%; width:50%; background:var(--accent); transition:width .25s; }
  #evaltext { font-family:var(--mono); font-size:.78rem; margin-top:6px; color:var(--muted); }

  #moves { font-family:var(--mono); font-size:.78rem; line-height:1.7;
           max-height:230px; overflow-y:auto; }
  #moves b { display:inline-block; width:26px; color:var(--muted); font-weight:400; }
  #status { font-family:var(--mono); font-size:.82rem; min-height:1.2em; }
  .thinking { color:var(--accent); }
  #promo { display:none; gap:6px; }
  #promo.on { display:flex; }
</style>
</head>
<body>
  <div>
    <h1 id="title">chessbot</h1>
    <div id="board"></div>
  </div>

  <div class="side">
    <div class="card">
      <div id="status">—</div>
      <div id="evalwrap" style="margin-top:8px"><div id="evalbar"></div></div>
      <div id="evaltext">no evaluation yet</div>
    </div>

    <div class="card" id="promo">
      <button data-p="q">♕ queen</button><button data-p="r">♖ rook</button>
      <button data-p="b">♗ bishop</button><button data-p="n">♘ knight</button>
    </div>

    <div class="card">
      <label for="gen">Generation</label>
      <select id="gen"></select>
      <label for="sims" style="margin-top:10px">Search — <span id="simsval">32</span> sims</label>
      <input type="range" id="sims" min="0" max="256" step="8" value="32">
      <label for="side" style="margin-top:10px">You play</label>
      <select id="side"><option value="w">White</option><option value="b">Black</option></select>
      <div class="row" style="margin-top:12px">
        <button id="new">New game</button><button id="undo">Undo</button>
      </div>
      <div class="row" style="margin-top:8px">
        <button id="flip">Flip board</button>
      </div>
    </div>

    <div class="card"><label>Moves</label><div id="moves"></div></div>
  </div>

<script>
const GLYPH = {P:"♙",N:"♘",B:"♗",R:"♖",Q:"♕",K:"♔",p:"♟",n:"♞",b:"♝",r:"♜",q:"♛",k:"♚"};
const FILES = "abcdefgh";
let state = null, sel = null, flipped = false, pending = null, busy = false;

const $ = (id) => document.getElementById(id);
const api = async (path, body) => {
  const r = await fetch(path, body ? {method:"POST", headers:{"Content-Type":"application/json"},
                                      body:JSON.stringify(body)} : {});
  if (!r.ok) throw new Error(await r.text());
  return r.json();
};

function fenToBoard(fen) {
  const rows = fen.split(" ")[0].split("/");
  const out = {};
  rows.forEach((row, i) => {
    let f = 0;
    for (const ch of row) {
      if (/\d/.test(ch)) { f += +ch; continue; }
      out[FILES[f] + (8 - i)] = ch;
      f++;
    }
  });
  return out;
}

function draw() {
  if (!state) return;
  const pieces = fenToBoard(state.fen);
  const board = $("board");
  board.innerHTML = "";
  const ranks = flipped ? [1,2,3,4,5,6,7,8] : [8,7,6,5,4,3,2,1];
  const files = flipped ? [...FILES].reverse() : [...FILES];

  for (const r of ranks) for (const f of files) {
    const name = f + r;
    const d = document.createElement("div");
    const dark = (FILES.indexOf(f) + r) % 2 === 0;
    d.className = "sq " + (dark ? "dark" : "light");
    if (state.last && state.last.includes(name)) d.classList.add("last");
    const p = pieces[name];
    if (p) {
      d.textContent = GLYPH[p];
      d.className += p === p.toUpperCase() ? " wp" : " bp";
      d.classList.add(dark ? "dark" : "light", "sq");
    }
    if (sel === name) d.classList.add("sel");
    if (sel && (state.legal[sel] || []).some(u => u.slice(2,4) === name)) {
      const dot = document.createElement("div");
      dot.className = "dot";
      if (p) d.classList.add("occupied");
      d.appendChild(dot);
    }
    d.onclick = () => clickSquare(name);
    board.appendChild(d);
  }

  $("moves").innerHTML = state.history.map((san, i) =>
    (i % 2 === 0 ? `<b>${i/2+1}.</b>` : "") + san + " ").join("");
  $("moves").scrollTop = $("moves").scrollHeight;

  $("status").textContent = busy ? "thinking…" : (state.status || (state.your_turn ? "your move" : "—"));
  $("status").className = busy ? "thinking" : "";
  $("title").textContent = "chessbot — gen" + String(state.gen).padStart(3,"0");

  if (state.eval === null) { $("evaltext").textContent = "no evaluation yet"; $("evalbar").style.width = "50%"; }
  else {
    // Shown from the human's point of view, not the engine's.
    const v = state.human_white === state.eval_white ? state.eval : -state.eval;
    $("evalbar").style.width = ((v + 1) / 2 * 100).toFixed(1) + "%";
    $("evaltext").textContent = "engine sees " + (state.eval >= 0 ? "+" : "") + state.eval.toFixed(2) +
      " for itself" + (Math.abs(state.eval) > .6 ? "  (confident)" : "");
  }
}

function clickSquare(name) {
  if (busy || !state || state.status || !state.your_turn) return;
  const targets = (state.legal[name] || []);
  if (sel && sel !== name) {
    const matches = (state.legal[sel] || []).filter(u => u.slice(2,4) === name);
    if (matches.length > 1) { pending = matches; sel = null; $("promo").classList.add("on"); draw(); return; }
    if (matches.length === 1) { sel = null; send(matches[0]); return; }
  }
  sel = targets.length ? name : null;
  draw();
}

async function send(uci) {
  busy = true; draw();
  try { state = await api("/api/move", {uci}); }
  catch (e) { $("status").textContent = "error: " + e.message; }
  busy = false; sel = null; draw();
}

$("promo").onclick = (e) => {
  const p = e.target.closest("button")?.dataset.p;
  if (!p || !pending) return;
  const uci = pending.find(u => u.endsWith(p)) || pending[0];
  pending = null; $("promo").classList.remove("on"); send(uci);
};

async function newGame() {
  busy = true; draw();
  state = await api("/api/new", {
    gen: +$("gen").value, sims: +$("sims").value, human_white: $("side").value === "w"
  });
  flipped = $("side").value === "b";
  busy = false; sel = null; draw();
}

$("new").onclick = newGame;
$("undo").onclick = async () => { state = await api("/api/undo", {}); sel = null; draw(); };
$("flip").onclick = () => { flipped = !flipped; draw(); };
$("sims").oninput = (e) => $("simsval").textContent = e.target.value;
$("sims").onchange = async (e) => { state = await api("/api/config", {sims:+e.target.value}); draw(); };

(async () => {
  const info = await api("/api/gens");
  $("gen").innerHTML = info.gens.map(g =>
    `<option value="${g}"${g === info.gens[info.gens.length-1] ? " selected" : ""}>gen ${String(g).padStart(3,"0")}</option>`
  ).join("");
  state = await api("/api/state");
  $("sims").value = state.sims; $("simsval").textContent = state.sims;
  draw();
})();
</script>
</body>
</html>
"""
