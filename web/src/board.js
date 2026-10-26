// A chess board widget, with no dependencies and no image assets.
//
// Pieces are the six solid Unicode glyphs (U+265A..U+265F) for *both* colours,
// separated by fill and outline rather than by using the outlined code points.
// The outlined glyphs render inconsistently across platforms -- on several they
// are a different weight, or simply missing -- whereas the solid six are
// reliable everywhere, and drawing white as "the same shape, inverted" makes
// the two sets match by construction.
//
// The widget knows nothing about chess. It is told which squares hold what,
// which moves are legal, and whose turn it is; it reports attempted moves.

const GLYPH = { p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚" };
const FILES = "abcdefgh";

export class Board {
  /** @param onMove called with a UCI move the user attempted. */
  constructor(root, { onMove }) {
    this.root = root;
    this.onMove = onMove;
    this.orientation = "white";
    this.pieces = new Map();   // square -> {color, role}
    this.dests = new Map();
    this.turn = "white";
    this.interactive = false;
    this.selected = null;
    this.lastMove = null;
    this.checkSquare = null;
    this.drag = null;
    this.squares = new Map();

    root.classList.add("board");
    this.#buildGrid();
    root.addEventListener("pointerdown", (e) => this.#down(e));
    window.addEventListener("pointermove", (e) => this.#move(e));
    window.addEventListener("pointerup", (e) => this.#up(e));
  }

  #buildGrid() {
    this.root.replaceChildren();
    for (let rank = 8; rank >= 1; rank--) {
      for (let f = 0; f < 8; f++) {
        const name = FILES[f] + rank;
        const sq = document.createElement("div");
        sq.className = `sq ${(f + rank) % 2 ? "light" : "dark"}`;
        sq.dataset.square = name;
        this.squares.set(name, sq);
        this.root.append(sq);
      }
    }
    this.#applyOrientation();
  }

  #applyOrientation() {
    this.root.classList.toggle("flipped", this.orientation === "black");
  }

  flip() { this.setOrientation(this.orientation === "white" ? "black" : "white"); }

  setOrientation(color) {
    this.orientation = color;
    this.#applyOrientation();
  }

  /** Replace the whole position. `fen` is a full FEN; only the board half is read. */
  setPosition(fen, { turn, dests, lastMove, check, interactive }) {
    this.pieces = parsePlacement(fen.split(" ")[0]);
    this.turn = turn;
    this.dests = dests ?? new Map();
    this.lastMove = lastMove ?? null;
    this.interactive = !!interactive;
    this.selected = null;
    this.checkSquare = null;
    if (check) {
      for (const [sq, p] of this.pieces) {
        if (p.role === "k" && p.color === turn) this.checkSquare = sq;
      }
    }
    this.render();
  }

  render() {
    for (const [name, sq] of this.squares) {
      sq.className = `sq ${(FILES.indexOf(name[0]) + Number(name[1])) % 2 ? "light" : "dark"}`;
      if (this.lastMove && (name === this.lastMove.slice(0, 2) || name === this.lastMove.slice(2, 4))) {
        sq.classList.add("last");
      }
      if (name === this.selected) sq.classList.add("sel");
      if (name === this.checkSquare) sq.classList.add("check");

      const piece = this.pieces.get(name);
      const canMove = this.interactive && this.dests.has(name);
      const target = this.selected && (this.dests.get(this.selected) ?? []).includes(name);

      sq.replaceChildren();
      if (piece) {
        const el = document.createElement("span");
        el.className = `piece ${piece.color}`;
        el.textContent = GLYPH[piece.role];
        if (canMove) el.classList.add("movable");
        if (this.drag?.from === name) el.classList.add("dragging");
        sq.append(el);
      }
      if (target) {
        const dot = document.createElement("span");
        dot.className = piece ? "capture" : "dot";
        sq.append(dot);
      }
    }
  }

  // --- pointer handling ---------------------------------------------------

  #squareAt(x, y) {
    const el = document.elementFromPoint(x, y);
    return el?.closest(".sq")?.dataset.square ?? null;
  }

  #down(e) {
    // A promotion menu is open and the click is inside it: leave it alone. The
    // menu answers on `click`, which lands after this, and anything that
    // re-renders here would tear the menu out of the DOM before it is read.
    if (e.target.closest?.(".promo")) return;
    if (!this.interactive || e.button !== 0) return;
    const name = this.#squareAt(e.clientX, e.clientY);
    if (!name) return;

    // Second click of a click-move, or a drop target chosen by clicking.
    if (this.selected && (this.dests.get(this.selected) ?? []).includes(name)) {
      this.#attempt(this.selected, name);
      return;
    }
    if (!this.dests.has(name)) {
      this.selected = null;
      this.render();
      return;
    }
    e.preventDefault();
    this.selected = name;
    this.drag = { from: name, ghost: this.#ghost(name, e) };
    this.render();
  }

  #ghost(name, e) {
    const piece = this.pieces.get(name);
    if (!piece) return null;
    const el = document.createElement("span");
    el.className = `piece ${piece.color} ghost`;
    el.textContent = GLYPH[piece.role];
    const size = this.root.getBoundingClientRect().width / 8;
    el.style.fontSize = `${size * 0.78}px`;
    el.style.width = el.style.height = `${size}px`;
    document.body.append(el);
    place(el, e.clientX, e.clientY);
    return el;
  }

  #move(e) {
    if (this.drag?.ghost) place(this.drag.ghost, e.clientX, e.clientY);
  }

  #up(e) {
    if (!this.drag) return;
    const { from, ghost } = this.drag;
    ghost?.remove();
    this.drag = null;
    const to = this.#squareAt(e.clientX, e.clientY);
    // A click rather than a drag: leave the piece selected so the next click
    // completes the move.
    if (!to || to === from) { this.render(); return; }
    if ((this.dests.get(from) ?? []).includes(to)) this.#attempt(from, to);
    else { this.selected = null; this.render(); }
  }

  #attempt(from, to) {
    this.selected = null;
    const piece = this.pieces.get(from);
    const promoting = piece?.role === "p" && (to[1] === "8" || to[1] === "1");
    // Render first. `render` rebuilds every square's children, so a menu opened
    // before it would be destroyed by it.
    this.render();
    if (promoting) this.#askPromotion(from, to, piece.color);
    else this.onMove(from + to);
  }

  /**
   * Promotion is four moves sharing a destination, so the board offers the
   * square once and asks which piece afterwards -- rather than auto-queening,
   * which quietly removes a real choice from the player.
   */
  #askPromotion(from, to, color) {
    const sq = this.squares.get(to);
    const menu = document.createElement("div");
    menu.className = "promo";
    for (const role of ["q", "r", "b", "n"]) {
      const b = document.createElement("button");
      b.className = `piece ${color}`;
      b.textContent = GLYPH[role];
      b.title = { q: "Queen", r: "Rook", b: "Bishop", n: "Knight" }[role];
      b.addEventListener("click", (ev) => {
        ev.stopPropagation();
        menu.remove();
        this.onMove(from + to + role);
      });
      menu.append(b);
    }
    sq.append(menu);
  }
}

function place(el, x, y) {
  el.style.left = `${x}px`;
  el.style.top = `${y}px`;
}

function parsePlacement(placement) {
  const out = new Map();
  const ranks = placement.split("/");
  for (let r = 0; r < 8; r++) {
    let f = 0;
    for (const ch of ranks[r]) {
      if (ch >= "1" && ch <= "8") { f += Number(ch); continue; }
      out.set(FILES[f] + (8 - r), {
        color: ch === ch.toUpperCase() ? "white" : "black",
        role: ch.toLowerCase(),
      });
      f++;
    }
  }
  return out;
}
