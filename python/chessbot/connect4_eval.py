"""Validation gates for the Connect4 loop.

Connect4 is here to make a broken RL loop *loud*. These are the checks that do
the shouting, ordered from cheapest to most demanding:

1. **Tactics** -- on positions with a move that wins on the spot, or a threat
   that must be blocked, does the raw policy pick it? Generated from real play
   and verified with the exact `is_winning_move`, so there is no dependence on
   the network being right about anything else.
2. **Opening** -- Connect4 is solved: the first player wins only by taking the
   centre column. A converging net puts its mass on column 3.
3. **Versus random** -- the floor. Anything that has learned at all wins nearly
   every game.
4. **Head to head** -- one net against another, which is what makes "is
   generation N better than N-1" a question with an answer.

All of these run on the *raw policy*, without search, unless told otherwise.
That is deliberate: search papers over a weak policy, and we want to see the
network itself improve.
"""

from __future__ import annotations

from dataclasses import dataclass

import chessbot_core as cc
import numpy as np
import torch

CENTRE = 3


def policy_probs(net, positions: list, device: str) -> np.ndarray:
    """Legal-masked policy probabilities for a list of positions."""
    if not positions:
        return np.zeros((0, cc.CONNECT4_POLICY_LEN), dtype=np.float32)
    obs = cc.connect4_encode_batch(positions)
    with torch.no_grad():
        logits, _ = net(torch.from_numpy(obs).to(device))
    logits = logits.float().cpu().numpy()

    mask = np.zeros_like(logits, dtype=bool)
    for i, p in enumerate(positions):
        mask[i, np.asarray(p.legal_moves(), dtype=np.intp)] = True
    logits = np.where(mask, logits, -np.inf)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def greedy_moves(net, positions: list, device: str) -> list[int]:
    return policy_probs(net, positions, device).argmax(axis=1).tolist()


# --- 1. tactical suite ------------------------------------------------------


@dataclass
class Tactic:
    position: cc.Connect4
    answer: int
    kind: str  # "win" or "block"


def build_tactics(count: int = 400, seed: int = 0) -> list[Tactic]:
    """Positions with a single forced correct move, found by random play.

    A position qualifies as a *win* if exactly one column completes a four for
    the mover. It qualifies as a *block* if the mover has no win but the
    opponent has exactly one, and that threat is not self-defeating to answer.
    """
    rng = np.random.default_rng(seed)
    out: list[Tactic] = []

    while len(out) < count:
        pos = cc.Connect4()
        while True:
            moves, outcome = pos.expand()
            if outcome is not None or not moves:
                break

            wins = [c for c in moves if pos.is_winning_move(c)]
            if len(wins) == 1:
                out.append(Tactic(pos, wins[0], "win"))
            elif not wins and len(moves) > 1:
                # No win of our own: is there exactly one reply that does not
                # hand the opponent an immediate win? Then it is forced.
                safe = [c for c in moves if not _opponent_wins_after(pos, c)]
                if len(safe) == 1:
                    out.append(Tactic(pos, safe[0], "block"))

            if len(out) >= count:
                break
            pos = pos.after(moves[rng.integers(len(moves))])

    return out[:count]


def _opponent_wins_after(pos: cc.Connect4, col: int) -> bool:
    """After we play `col`, can the opponent win immediately?"""
    nxt = pos.after(col)
    if nxt.outcome() is not None:
        return False
    return any(nxt.is_winning_move(c) for c in nxt.legal_moves())


def tactic_accuracy(net, tactics: list[Tactic], device: str) -> dict[str, float]:
    if not tactics:
        return {"win": float("nan"), "block": float("nan"), "all": float("nan")}
    picks = greedy_moves(net, [t.position for t in tactics], device)
    by_kind: dict[str, list[bool]] = {"win": [], "block": []}
    for t, pick in zip(tactics, picks):
        by_kind[t.kind].append(pick == t.answer)
    allv = by_kind["win"] + by_kind["block"]
    return {
        "win": float(np.mean(by_kind["win"])) if by_kind["win"] else float("nan"),
        "block": float(np.mean(by_kind["block"])) if by_kind["block"] else float("nan"),
        "all": float(np.mean(allv)) if allv else float("nan"),
    }


# --- 2. opening preference --------------------------------------------------


def centre_preference(net, device: str) -> float:
    """Probability mass the net puts on the (uniquely winning) centre column."""
    return float(policy_probs(net, [cc.Connect4()], device)[0, CENTRE])


# --- 3 & 4. match play ------------------------------------------------------


def play_match(
    net_a,
    net_b,
    games: int,
    device: str,
    seed: int = 0,
    sims: int = 0,
    opponent_random: bool = False,
    opening_plies: int = 2,
) -> dict[str, float]:
    """Play `games` games, net_a taking first move in half of them.

    `sims = 0` means raw policy. Anything higher runs a real search for both
    sides, which is slower but measures what the engine would actually play.

    `opening_plies` random moves are played before either net takes over. This
    is not optional decoration: greedy policy play is deterministic, so without
    randomised openings a 200-game match is two distinct games repeated 100
    times each, and the score it reports is noise dressed as data.
    """
    rng = np.random.default_rng(seed)
    boards = [cc.Connect4() for _ in range(games)]

    for i in range(games):
        for _ in range(opening_plies):
            moves, outcome = boards[i].expand()
            if outcome is not None or not moves:
                break
            boards[i] = boards[i].after(int(moves[rng.integers(len(moves))]))
    # a_first[i] decides who moves first, so a first-player advantage cannot be
    # mistaken for one net being stronger.
    a_first = np.arange(games) % 2 == 0
    results = np.zeros(games)  # +1 a wins, -1 b wins, 0 draw
    done = np.zeros(games, dtype=bool)

    while not done.all():
        live = np.where(~done)[0]
        # Whose turn is it in each live game?
        plies = np.array([boards[i].plies for i in live])
        a_to_move = (plies % 2 == 0) == a_first[live]

        for is_a in (True, False):
            idx = live[a_to_move == is_a]
            if len(idx) == 0:
                continue
            positions = [boards[i] for i in idx]

            if is_a is False and opponent_random:
                picks = [
                    int(p.legal_moves()[rng.integers(len(p.legal_moves()))])
                    for p in positions
                ]
            else:
                net = net_a if is_a else net_b
                picks = (
                    search_moves(net, positions, device, sims, int(rng.integers(1 << 30)))
                    if sims > 0
                    else greedy_moves(net, positions, device)
                )

            for i, mv in zip(idx, picks):
                boards[i] = boards[i].after(int(mv))

        for i in live:
            outcome = boards[i].outcome()
            if outcome is None:
                continue
            done[i] = True
            if outcome == "draw":
                results[i] = 0.0
            else:
                # The side to move has lost, so the previous mover won.
                loser_is_first = boards[i].plies % 2 == 0
                winner_is_a = (not loser_is_first) == a_first[i]
                results[i] = 1.0 if winner_is_a else -1.0

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


def search_moves(net, positions, device, sims, seed) -> list[int]:
    search = cc.Connect4Search(positions, sims=sims, max_considered=7, seed=seed)
    while (obs := search.next_batch()) is not None:
        with torch.no_grad():
            logits, wdl = net(torch.from_numpy(obs).to(device))
            probs = torch.softmax(wdl.float(), dim=-1)
            values = (probs[:, 0] - probs[:, 2]).cpu().numpy()
        search.submit(
            np.ascontiguousarray(logits.float().cpu().numpy()),
            np.ascontiguousarray(values.astype(np.float32)),
        )
    return search.moves()


def elo_from_score(score: float) -> float:
    """Elo difference implied by a score in (0, 1)."""
    score = min(max(score, 1e-4), 1 - 1e-4)
    return -400.0 * np.log10(1.0 / score - 1.0)
