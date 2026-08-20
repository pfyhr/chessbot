"""Validation gates for the chess loop.

Same principle as Connect4: prefer checks with *exact* ground truth over checks
that merely correlate with strength, because only the former can tell you the
loop is broken rather than slow.

Chess has no solved opening, so there is no direct analogue of Connect4's centre
column. What it does have is forced mates, which are exact and cheap to generate:

1. **mate-in-1** -- a mate is available; does the policy play it?
2. **avoid-mate-in-1** -- the opponent mates next move unless this one prevents
   it; does the policy defend? The symmetric test, and the harder one.
3. **vs random** -- the floor.
4. **vs previous** -- is generation N better than N-1.
5. **opening** -- mass on the four sound first moves. Not ground truth, but a
   cheap read on whether the policy has learned anything about openings at all.

Suites are generated with the Rust core rather than python-chess: finding mates
means testing every legal move in every position, and doing that in Python is
minutes rather than seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import chessbot_core as cc
import numpy as np
import torch

SOUND_OPENINGS = ("e2e4", "d2d4", "g1f3", "c2c4")


# --- policy over positions --------------------------------------------------


def policy_probs(net, positions: list, device: str, batch: int = 512) -> np.ndarray:
    """Legal-masked policy probabilities, batched."""
    if not positions:
        return np.zeros((0, cc.POLICY_LEN), dtype=np.float32)

    out = np.zeros((len(positions), cc.POLICY_LEN), dtype=np.float32)
    for start in range(0, len(positions), batch):
        chunk = positions[start : start + batch]
        obs = cc.encode_batch(chunk)
        with torch.no_grad():
            logits, _ = net(torch.from_numpy(obs).to(device))
        logits = logits.float().cpu().numpy()

        mask = cc.legal_mask_batch(chunk)
        logits = np.where(mask, logits, -np.inf)
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        out[start : start + len(chunk)] = e / e.sum(axis=1, keepdims=True)
    return out


def _index_to_uci(pos, index: int) -> str | None:
    for m in pos.legal_moves():
        if pos.policy_index(m) == index:
            return m
    return None


def best_moves(net, positions: list, device: str) -> list[str]:
    probs = policy_probs(net, positions, device)
    return [_index_to_uci(p, int(probs[i].argmax())) for i, p in enumerate(positions)]


# --- suites -----------------------------------------------------------------


@dataclass
class Tactic:
    fen: str
    answers: tuple[str, ...]
    kind: str

    @property
    def position(self) -> cc.Position:
        return cc.Position.from_fen(self.fen)


def _mates_available(pos) -> list[str]:
    """Moves that deliver immediate checkmate."""
    out = []
    for m in pos.legal_moves():
        nxt = pos.after(m)
        moves, outcome = nxt.expand()
        # The side to move next has lost and has no moves -- that is mate, not a
        # draw by rule.
        if outcome == "loss" and not moves:
            out.append(m)
    return out


def build_tactics(
    mates: int = 250,
    defences: int = 250,
    seed: int = 0,
    max_safe_fraction: float = 0.34,
) -> list[Tactic]:
    """Find forced-mate and forced-defence positions by random play.

    `max_safe_fraction` is what makes the defence suite a real test. Without it,
    almost any position where *some* reply allows mate qualifies -- and in most of
    those the large majority of replies are safe, so a random mover scores 0.75
    and the metric has nowhere to go. Keeping only positions where at most a third
    of replies survive puts the random baseline near 0.2 and leaves headroom for
    the network to show improvement.

    Call `baselines()` on the result; the random-mover score should always be
    reported next to the accuracy.
    """
    rng = np.random.default_rng(seed)
    found: list[Tactic] = []
    n_mate = n_def = 0

    while n_mate < mates or n_def < defences:
        pos = cc.Position()
        for _ in range(220):
            moves, outcome = pos.expand()
            if outcome is not None or not moves:
                break

            if n_mate < mates:
                winning = _mates_available(pos)
                if winning:
                    found.append(Tactic(pos.fen(), tuple(winning), "mate"))
                    n_mate += 1

            if n_def < defences:
                # Which replies allow the opponent to mate immediately?
                unsafe = 0
                safe: list[str] = []
                for m in moves:
                    nxt = pos.after(m)
                    if nxt.outcome() is not None:
                        safe.append(m)
                        continue
                    if _mates_available(nxt):
                        unsafe += 1
                    else:
                        safe.append(m)
                # A real defensive test: most replies must lose, or the position
                # is passed by accident rather than by defending.
                if unsafe > 0 and safe and len(safe) / len(moves) <= max_safe_fraction:
                    found.append(Tactic(pos.fen(), tuple(safe), "defend"))
                    n_def += 1

            pos = pos.after(moves[rng.integers(len(moves))])
            if n_mate >= mates and n_def >= defences:
                break

    return found


def baselines(tactics: list[Tactic]) -> dict[str, float]:
    """What a uniformly random legal move would score on each suite.

    Report this beside the accuracy. An unlabelled 0.57 reads as progress; next to
    a 0.75 baseline it reads as what it is.
    """
    out = {}
    for kind in ("mate", "defend"):
        rows = [t for t in tactics if t.kind == kind]
        out[kind] = (
            float(np.mean([len(t.answers) / len(t.position.legal_moves()) for t in rows]))
            if rows
            else float("nan")
        )
    return out


def tactic_accuracy(net, tactics: list[Tactic], device: str) -> dict[str, float]:
    if not tactics:
        return {"mate": float("nan"), "defend": float("nan"), "all": float("nan")}

    positions = [t.position for t in tactics]
    picks = best_moves(net, positions, device)
    by_kind: dict[str, list[bool]] = {"mate": [], "defend": []}
    for t, pick in zip(tactics, picks):
        by_kind[t.kind].append(pick in t.answers)

    everything = by_kind["mate"] + by_kind["defend"]
    return {
        "mate": float(np.mean(by_kind["mate"])) if by_kind["mate"] else float("nan"),
        "defend": float(np.mean(by_kind["defend"])) if by_kind["defend"] else float("nan"),
        "all": float(np.mean(everything)) if everything else float("nan"),
    }


def opening_mass(net, device: str) -> float:
    """Policy mass on e4, d4, Nf3 and c4 from the initial position."""
    start = cc.Position()
    probs = policy_probs(net, [start], device)[0]
    return float(sum(probs[start.policy_index(m)] for m in SOUND_OPENINGS))


# --- match play -------------------------------------------------------------


def search_moves(net, positions, device, sims, seed) -> list[str]:
    search = cc.ChessSearch(positions, sims=sims, max_considered=16, seed=seed)
    while (obs := search.next_batch()) is not None:
        with torch.no_grad():
            logits, wdl = net(torch.from_numpy(obs).to(device))
            p = torch.softmax(wdl.float(), dim=-1)
            values = (p[:, 0] - p[:, 2]).cpu().numpy()
        search.submit(
            np.ascontiguousarray(logits.float().cpu().numpy()),
            np.ascontiguousarray(values.astype(np.float32)),
        )
    return search.moves()


def play_match(
    net_a,
    net_b,
    games: int,
    device: str,
    seed: int = 0,
    sims: int = 0,
    opponent_random: bool = False,
    opening_plies: int = 4,
    max_plies: int = 200,
) -> dict[str, float]:
    """Play `games` games, net_a taking White in half.

    Openings are randomised: greedy policy play is deterministic, so without that
    every game would be a replay of the same two.
    """
    rng = np.random.default_rng(seed)
    boards = [cc.Position() for _ in range(games)]
    for i in range(games):
        for _ in range(opening_plies):
            moves, outcome = boards[i].expand()
            if outcome is not None or not moves:
                break
            boards[i] = boards[i].after(moves[rng.integers(len(moves))])

    a_white = np.arange(games) % 2 == 0
    results = np.zeros(games)
    done = np.zeros(games, dtype=bool)
    plies = np.zeros(games, dtype=int)

    while not done.all():
        live = np.where(~done)[0]
        a_to_move = np.array([boards[i].turn == "white" for i in live]) == a_white[live]

        for is_a in (True, False):
            idx = live[a_to_move == is_a]
            if len(idx) == 0:
                continue
            positions = [boards[i] for i in idx]

            if not is_a and opponent_random:
                picks = [
                    p.legal_moves()[rng.integers(len(p.legal_moves()))] for p in positions
                ]
            else:
                net = net_a if is_a else net_b
                picks = (
                    search_moves(net, positions, device, sims, int(rng.integers(1 << 30)))
                    if sims > 0
                    else best_moves(net, positions, device)
                )

            for i, mv in zip(idx, picks):
                boards[i] = boards[i].after(mv)
                plies[i] += 1

        for i in live:
            outcome = boards[i].outcome()
            if outcome is None and plies[i] < max_plies:
                continue
            done[i] = True
            if outcome is None or outcome == "draw":
                results[i] = 0.0
            else:
                # Side to move has lost, so the previous mover won.
                loser_is_white = boards[i].turn == "white"
                results[i] = 1.0 if (not loser_is_white) == a_white[i] else -1.0

    wins = int((results > 0).sum())
    losses = int((results < 0).sum())
    draws = int((results == 0).sum())
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": (wins + 0.5 * draws) / games,
    }


def elo_from_score(score: float) -> float:
    score = min(max(score, 1e-4), 1 - 1e-4)
    return -400.0 * np.log10(1.0 / score - 1.0)
