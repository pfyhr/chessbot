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


def test_packed_batches_are_the_dense_batches(net):
    """Same seed, same answers: every packed batch must unpack to the dense one."""
    dense = cc.ChessSelfPlay(concurrency=5, total_games=5, seed=11, **KW)
    packed = cc.ChessSelfPlay(concurrency=5, total_games=5, seed=11, **KW)
    batches = 0
    while (obs := dense.next_batch()) is not None:
        bits, scale, legal = packed.next_batch_packed()
        planes = pipeline.unpack_obs(torch.from_numpy(bits.view(np.int64)),
                                     torch.from_numpy(scale), cc.OBS_SHAPE).numpy()
        np.testing.assert_array_equal(planes, obs)
        with torch.no_grad():
            logits, wdl = net(torch.from_numpy(obs))
        values = np.ascontiguousarray(wdl_to_scalar(wdl.float()).numpy().astype(np.float32))
        compact = logits.gather(1, torch.from_numpy(legal).long())
        dense.submit(np.ascontiguousarray(logits.numpy()), values)
        packed.submit_packed(np.ascontiguousarray(compact.numpy()), values)
        batches += 1
    assert packed.next_batch_packed() is None
    assert batches > 5


def test_packed_pipeline_reproduces_the_legacy_loop(net):
    ref, ref_stats = legacy_loop(net, games=6, concurrency=4, seed=7)
    pools = pipeline.make_pools(pools=1, concurrency=4, games=6, seed=7, **KW)
    data, stats = pipeline.merge(pipeline.run(pools, pipeline.make_forward(net, "cpu", packed=True)))
    for a, b in zip(ref, data):
        np.testing.assert_array_equal(a, b)
    assert stats == ref_stats


def test_legal_rows_hold_exactly_the_legal_moves():
    sp = cc.ChessSelfPlay(concurrency=3, total_games=3, seed=2, **KW)
    _bits, _scale, legal = sp.next_batch_packed()
    start = sorted(cc.Position().policy_index(m) for m in cc.Position().legal_moves())
    for row in legal:  # every game starts from the initial position
        assert sorted(row[:20].tolist()) == start
        assert (row[20:] == 0).all()


def _devices():
    out = ["cpu"]
    if torch.backends.mps.is_available():
        out.append("mps")
    if torch.cuda.is_available():
        out.append("cuda")
    return out


@pytest.mark.parametrize("device", _devices())
def test_unpack_obs_is_exact_on_every_device(device):
    """int64 shifts and `& 1` must be exact wherever the packed bridge runs.

    Masks are random with bit 63 set on about half of them, the case an
    arithmetic shift could get wrong; the reference is a plain numpy unpack.
    """
    rng = np.random.default_rng(0)
    planes, h, w = cc.OBS_SHAPE
    bits = rng.integers(0, 2**63, size=(64, planes), dtype=np.int64)
    bits[rng.random(bits.shape) < 0.5] |= np.int64(-2**63)
    scale = rng.random((64, planes)).astype(np.float32)
    u = bits.view(np.uint64)[..., None] >> np.arange(h * w, dtype=np.uint64)
    ref = ((u & np.uint64(1)).astype(np.float32) * scale[..., None]).reshape(64, planes, h, w)
    got = pipeline.unpack_obs(torch.from_numpy(bits).to(device),
                              torch.from_numpy(scale).to(device), cc.OBS_SHAPE)
    np.testing.assert_array_equal(got.cpu().numpy(), ref)
