"""Pooled self-play: one pool must be the old loop exactly, several must add up."""

import chessbot_core as cc
import numpy as np
import pytest
import torch

from chessbot import pipeline
from chessbot.net import Net, wdl_to_scalar

KW = dict(sims=8, max_considered=4, max_plies=12)


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    planes, h, w = cc.OBS_SHAPE
    return Net(planes, (h, w), cc.POLICY_LEN, 1, 8).eval()


def legacy_loop(net, games, concurrency, seed):
    """The single-driver loop as it was before pools existed."""
    sp = cc.ChessSelfPlay(concurrency=concurrency, total_games=games, seed=seed, **KW)
    while (obs := sp.next_batch()) is not None:
        with torch.no_grad():
            logits, wdl = net(torch.from_numpy(obs))
        sp.submit(np.ascontiguousarray(logits.float().numpy()),
                  np.ascontiguousarray(wdl_to_scalar(wdl.float()).numpy().astype(np.float32)))
    return sp.take_training_data(), dict(sp.stats())


def test_split_is_even_and_exact():
    assert pipeline.split(10, 3) == [4, 3, 3]
    assert sum(pipeline.split(1023, 4)) == 1023


def test_one_pool_reproduces_the_legacy_loop(net):
    (ref, ref_stats) = legacy_loop(net, games=6, concurrency=4, seed=7)
    pools = pipeline.make_pools(pools=1, concurrency=4, games=6, seed=7, **KW)
    data, stats = pipeline.merge(pipeline.run(pools, pipeline.make_forward(net, "cpu")))
    for a, b in zip(ref, data):
        np.testing.assert_array_equal(a, b)
    assert stats == ref_stats


def test_two_pools_play_every_game_and_merge_consistently(net):
    pools = pipeline.make_pools(pools=2, concurrency=4, games=7, seed=3, **KW)
    results = pipeline.run(pools, pipeline.make_forward(net, "cpu", bucket=4, cap=2))
    data, stats = pipeline.merge(results)
    assert stats["games"] == 7
    n = len(data[0])
    assert n == sum(len(r.data[0]) for r in results) > 0
    assert all(len(a) == n for a in data)
    assert stats["evaluations"] == sum(r.timing.evals for r in results)


def test_too_many_pools_is_rejected():
    with pytest.raises(ValueError):
        pipeline.make_pools(pools=5, concurrency=4, games=8, seed=0, **KW)
