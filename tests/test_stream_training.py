"""One-pass streaming, sampler workers, and the WSD trainer with branches."""

import json
import os
import random

import numpy as np

from chessbot import gamestore, lc0_data, train_supervised
from test_lc0_data import _random_game


class Recording(gamestore.StreamSampler):
    def __init__(self, *a, **k):
        self.picked = []
        super().__init__(*a, **k)

    def _pick(self):
        s, g = super()._pick()
        self.picked.append((os.path.basename(s.path), g))
        return s, g


def _stream(tmp_path, n_shards=4, games=5, listed=None):
    rng = random.Random(11)
    tables = lc0_data.Tables()
    names = []
    for k in range(n_shards):
        gs = [lc0_data.convert_game(_random_game(rng)[0], tables) for _ in range(games)]
        gamestore.write_shard(str(tmp_path / f"s{k}"), gs)
        names.append(f"s{k}")
    (tmp_path / "order.txt").write_text("\n".join(listed or names) + "\n")
    return names


def test_every_game_exactly_once_and_shards_retired(tmp_path):
    names = _stream(tmp_path)
    s = Recording(str(tmp_path), window=2, capacity=64, seed=0, poll=0.01)
    while len(s.picked) < 4 * 5:
        s._pick()
    assert len(set(s.picked)) == 20
    assert sorted((tmp_path / "consumed-0.txt").read_text().split()) == names


def test_parts_are_disjoint_and_restart_skips_consumed(tmp_path):
    _stream(tmp_path)
    a = Recording(str(tmp_path), window=1, capacity=16, seed=0, part=(0, 2), poll=0.01)
    while len(a.picked) < 10:
        a._pick()
    assert {n for n, _ in a.picked} == {"s0", "s2"}
    b = Recording(str(tmp_path), window=1, capacity=16, seed=0, part=(1, 2), poll=0.01)
    while len(b.picked) < 10:
        b._pick()
    assert {n for n, _ in b.picked} == {"s1", "s3"}
    assert sorted((tmp_path / "consumed-0.txt").read_text().split()) == ["s0", "s2"]
    # Restarted, part 0 skips what it consumed and goes on to new data.
    gamestore.write_shard(str(tmp_path / "s4"), [lc0_data.convert_game(
        _random_game(random.Random(5))[0], lc0_data.Tables())])
    (tmp_path / "order.txt").write_text("s0\ns1\ns2\ns3\ns4\n")
    c = Recording(str(tmp_path), window=1, capacity=4, seed=0, part=(0, 2), poll=0.01)
    assert {n for n, _ in c.picked} == {"s4"}


def test_waits_for_a_listed_shard_still_being_copied(tmp_path):
    names = _stream(tmp_path, n_shards=2)
    os.rename(tmp_path / "s1", tmp_path / "s1.away")
    s = Recording(str(tmp_path), window=2, capacity=4, seed=0, poll=0.01)
    while len(s.picked) < 5:
        s._pick()
    assert {n for n, _ in s.picked} == {"s0"}
    os.rename(tmp_path / "s1.away", tmp_path / "s1")
    while len(s.picked) < 10:
        s._pick()
    assert {n for n, _ in s.picked} == set(names)


def test_workers_deliver_batches(tmp_path):
    from chessbot.batch_workers import WorkerBatches
    _stream(tmp_path)
    wb = WorkerBatches(str(tmp_path), workers=2, batch=32, capacity=64, window=1)
    try:
        b = wb.batch()
        assert b["bits"].dtype == np.int64 and b["bits"].shape[0] == 32
        np.testing.assert_allclose(b["p"].sum(1), 1.0, atol=1e-2)
    finally:
        wb.close()


def test_wsd_trainer_writes_annealed_branches_and_resumes(tmp_path):
    data = tmp_path / "stream"
    data.mkdir()
    _stream(data, n_shards=6, games=8)
    val = tmp_path / "val"
    _stream(val, n_shards=1, games=4)
    out = tmp_path / "out"
    args = ["--stream", str(data), "--val", str(val / "s0"), "--out", str(out),
            "--blocks", "1", "--channels", "8", "--batch", "64", "--buffer", "256",
            "--schedule", "wsd", "--warmup", "2", "--lr", "1e-3",
            "--branch-at", "640,1280", "--branch-frac", "0.5",
            "--eval-every", "5", "--save-every", "4", "--device", "cpu"]
    train_supervised.main(args)
    rows = [json.loads(l) for l in open(out / "log.jsonl")]
    branches = [r for r in rows if r["kind"] == "branch"]
    assert [r["branch"] for r in branches] == [640, 1280]
    assert branches[0]["positions"] >= 640 + 320 and branches[1]["positions"] >= 1280 + 640
    assert (out / "branch_640.pt").exists() and (out / "branch_1280.pt").exists()
    assert train_supervised.branch_lr(960, 640, 320, 1e-3) == 0.0

    # Resuming a finished run does nothing more.
    train_supervised.main(args + ["--resume"])
    rows2 = [json.loads(l) for l in open(out / "log.jsonl")]
    assert len([r for r in rows2 if r["kind"] == "branch"]) == 2
