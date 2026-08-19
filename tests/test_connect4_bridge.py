"""Tests for the Connect4 game, search and self-play bindings.

The Rust side has its own unit tests; these check that the Python surface says
the same things, and that the self-play protocol behaves the way the training
loop assumes it does.
"""

from __future__ import annotations

import chessbot_core as cc
import numpy as np
import pytest


def test_constants():
    assert cc.CONNECT4_POLICY_LEN == 7
    assert cc.CONNECT4_OBS_SHAPE == (2, 6, 7)


def test_opening_position():
    p = cc.Connect4()
    assert sorted(p.legal_moves()) == [0, 1, 2, 3, 4, 5, 6]
    assert p.outcome() is None
    assert p.plies == 0
    assert p.turn == "first"


def test_legal_moves_are_indexable_ints():
    """PyO3 maps Vec<u8> to bytes, which breaks numpy indexing silently."""
    moves = cc.Connect4().legal_moves()
    assert isinstance(moves, list)
    assert all(isinstance(m, int) for m in moves)
    mask = np.zeros((1, 7), dtype=bool)
    mask[0, np.asarray(moves, dtype=np.intp)] = True
    assert mask.sum() == 7


def test_vertical_win_is_detected():
    p = cc.Connect4.from_moves([3, 4, 3, 4, 3, 4, 3])
    assert p.outcome() == "loss", "side to move has just been beaten"


def test_full_column_is_unplayable():
    p = cc.Connect4()
    for _ in range(6):
        p = p.after(0)
    assert 0 not in p.legal_moves()
    with pytest.raises(ValueError):
        p.play(0)


def test_is_winning_move_matches_playing_it():
    rng = np.random.default_rng(11)
    for _ in range(200):
        p = cc.Connect4()
        while True:
            moves, outcome = p.expand()
            if outcome is not None or not moves:
                break
            for c in moves:
                predicted = p.is_winning_move(c)
                assert predicted == (p.after(c).outcome() == "loss"), p.render()
            p = p.after(int(moves[rng.integers(len(moves))]))


def test_encode_batch_matches_individual():
    rng = np.random.default_rng(3)
    positions = []
    p = cc.Connect4()
    for _ in range(20):
        positions.append(p)
        moves, outcome = p.expand()
        if outcome is not None or not moves:
            break
        p = p.after(int(moves[rng.integers(len(moves))]))

    batch = cc.connect4_encode_batch(positions)
    assert batch.shape == (len(positions), 2, 6, 7)
    assert batch.dtype == np.float32
    for i, q in enumerate(positions):
        np.testing.assert_array_equal(batch[i], q.encode())


def test_encoding_is_from_the_movers_perspective():
    p = cc.Connect4.from_moves([3])
    obs = p.encode()
    # One stone on the board and it belongs to the opponent now.
    assert obs[0].sum() == 0.0
    assert obs[1].sum() == 1.0
    assert obs[1, 0, 3] == 1.0


# --- self-play driver -------------------------------------------------------


def run_selfplay(concurrency=16, total=32, sims=16):
    sp = cc.Connect4SelfPlay(
        concurrency=concurrency, total_games=total, sims=sims, seed=5
    )
    batches = []
    while (obs := sp.next_batch()) is not None:
        batches.append(obs.shape[0])
        n = obs.shape[0]
        sp.submit(
            np.zeros((n, 7), dtype=np.float32),
            np.zeros(n, dtype=np.float32),
        )
    return sp, batches


def test_selfplay_completes():
    sp, batches = run_selfplay()
    assert sp.is_done()
    assert sp.games_completed == 32
    assert max(batches) == 16


def test_batches_stay_full_until_the_tail():
    """Finished games are replaced immediately, so the batch should not sag.

    Only the last `concurrency` games have nothing to refill from, so the batch
    holds at the limit until that tail. Measured over the first 80% of the run,
    where refills are still available.
    """
    sp, batches = run_selfplay(concurrency=16, total=160, sims=8)
    assert sp.games_completed == 160

    body = batches[: int(len(batches) * 0.8)]
    assert all(b == 16 for b in body), (
        f"batch sagged mid-run: min {min(body)} of 16"
    )


def test_training_data_shapes_line_up():
    sp, _ = run_selfplay()
    obs, policy, z = sp.take_training_data()
    assert obs.shape[1:] == (2, 6, 7)
    assert policy.shape == (obs.shape[0], 7)
    assert z.shape == (obs.shape[0],)
    assert obs.dtype == policy.dtype == z.dtype == np.float32


def test_policy_targets_are_distributions():
    sp, _ = run_selfplay()
    _, policy, _ = sp.take_training_data()
    sums = policy.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-3)
    assert (policy >= 0).all()


def test_results_are_in_range_and_alternate():
    sp, _ = run_selfplay()
    _, _, z = sp.take_training_data()
    assert set(np.unique(z)).issubset({-1.0, 0.0, 1.0})


def test_stats_do_not_consume_training_data():
    """Reading stats must not drain the buffer, or the loop silently loses games."""
    sp, _ = run_selfplay()
    first = sp.stats()
    second = sp.stats()
    assert first["games"] == second["games"] == 32
    obs, _, _ = sp.take_training_data()
    assert obs.shape[0] > 0
    assert sp.stats()["games"] == 32, "stats changed after draining"


def test_stats_account_for_every_game():
    sp, _ = run_selfplay()
    s = sp.stats()
    assert s["first_wins"] + s["second_wins"] + s["draws"] == s["games"]
    assert 0 < s["mean_plies"] <= 42


# --- batched search ---------------------------------------------------------


def test_search_finds_an_immediate_win_with_a_flat_network():
    """Search quality, isolated from learning: uniform policy, zero value."""
    pos = cc.Connect4.from_moves([3, 0, 3, 1, 3, 2])
    assert pos.is_winning_move(3)

    found = 0
    trials = 12
    for seed in range(trials):
        s = cc.Connect4Search([pos], sims=64, max_considered=7, seed=seed)
        while (obs := s.next_batch()) is not None:
            n = obs.shape[0]
            s.submit(np.zeros((n, 7), np.float32), np.zeros(n, np.float32))
        if s.moves()[0] == 3:
            found += 1
    assert found == trials, f"missed the win in {trials - found}/{trials} searches"


def test_search_policies_are_distributions():
    positions = [cc.Connect4(), cc.Connect4.from_moves([3, 3])]
    s = cc.Connect4Search(positions, sims=32, seed=1)
    while (obs := s.next_batch()) is not None:
        n = obs.shape[0]
        s.submit(np.zeros((n, 7), np.float32), np.zeros(n, np.float32))
    pol = s.policies()
    assert pol.shape == (2, 7)
    np.testing.assert_allclose(pol.sum(axis=1), 1.0, atol=1e-3)
    assert s.values().shape == (2,)
