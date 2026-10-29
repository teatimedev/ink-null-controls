import json

import numpy as np
import pytest

from inknull import cli, metrics, nulls


@pytest.fixture
def stack():
    rng = np.random.default_rng(0)
    a = rng.integers(40, 200, size=(12, 256, 320), dtype=np.uint8)
    a[:, :, :16] = 0  # a no-data margin
    return a


def test_constant_keeps_mask(stack):
    out, p = nulls.constant(stack)
    valid = nulls.valid_mask(stack)
    assert (out[:, ~valid] == 0).all()
    assert (out[:, valid] == p["value"]).all()


def test_phase_scramble_keeps_histograms(stack):
    out, _ = nulls.phase_scramble(stack, seed=1)
    valid = nulls.valid_mask(stack)
    for k in range(stack.shape[0]):
        assert np.array_equal(np.sort(out[k][valid]), np.sort(stack[k][valid]))
    assert not np.array_equal(out, stack)
    assert (out[:, ~valid] == 0).all()


def test_phase_scramble_destroys_a_stroke():
    a = np.full((8, 128, 128), 100, np.uint8)
    a[:, 60:68, 20:108] = 180  # one bright horizontal stroke on every layer
    out, _ = nulls.phase_scramble(a, seed=3)
    row_profile = out[4].astype(float).mean(axis=1)
    assert row_profile[60:68].mean() - row_profile.mean() < 0.5 * (180 - 100)


def test_block_shuffle_round_trip(stack):
    out, p = nulls.block_shuffle(stack, block=32, seed=2)
    assert sorted(map(tuple, out.reshape(out.shape[0], -1).T.tolist())) == \
        sorted(map(tuple, stack.reshape(stack.shape[0], -1).T.tolist()))
    pred = out[3].astype(np.float32)
    assert np.array_equal(nulls.unshuffle_map(pred, p), stack[3].astype(np.float32))


def test_depth_nulls(stack):
    out, p = nulls.depth_shuffle(stack, seed=4)
    assert np.array_equal(out, stack[p["permutation"]])
    rev, _ = nulls.depth_reverse(stack)
    assert np.array_equal(rev[0], stack[-1])


def test_auc_matches_bruteforce():
    rng = np.random.default_rng(5)
    g = rng.integers(0, 256, size=4000).astype(np.uint8)
    lab = rng.integers(0, 2, size=4000).astype(bool)
    pos, neg = g[lab].astype(int), g[~lab].astype(int)
    brute = (np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg))
    assert metrics.auc(g, lab, ~lab) == pytest.approx(brute, abs=1e-12)


def test_threshold_for_recall():
    g = np.arange(100, dtype=np.uint8)
    ink = np.ones(100, bool)
    t = metrics.threshold_for_recall(g, ink, 0.8)
    assert np.mean(g[ink] >= t) >= 0.8 and np.mean(g[ink] >= t + 1) < 0.8


def test_score_flags_biased_model():
    rng = np.random.default_rng(6)
    labels = (rng.random((200, 200)) < 0.2).astype(np.int8)
    real = np.where(labels == 1, 0.6, 0.15) + rng.normal(0, 0.05, (200, 200))
    biased_constant = np.full((200, 200), 0.58)  # a model that outputs 0.58 on blank input
    r = metrics.score(real, {"constant": biased_constant}, labels=labels)
    assert any("structure-free" in f for f in r["flags"])
    assert r["null_threshold"]["recall"] < r["null_threshold"]["recall_at_calibration"]


def test_cli_make_and_score(tmp_path, stack):
    np.save(tmp_path / "stack.npy", stack)
    assert cli.main(["make", str(tmp_path / "stack.npy"), str(tmp_path / "nulls"), "--block", "32"]) == 0
    manifest = json.loads((tmp_path / "nulls" / "nulls.json").read_text())
    assert set(manifest["nulls"]) == set(nulls.ALL)
    rng = np.random.default_rng(7)
    np.save(tmp_path / "real.npy", rng.random(stack.shape[1:]).astype(np.float32))
    maps = tmp_path / "maps"
    maps.mkdir()
    for name in nulls.ALL:
        np.save(maps / f"{name}.npy", rng.random(stack.shape[1:]).astype(np.float32) * 0.1)
    code = cli.main(["score", "--real", str(tmp_path / "real.npy"), "--null-dir", str(maps),
                     "--manifest", str(tmp_path / "nulls" / "nulls.json"), "--out", str(tmp_path / "r.json"),
                     "--png", str(tmp_path / "p.png")])
    assert code == 0
    assert (tmp_path / "p.png").exists()


def test_pixel_shuffle_keeps_depth_profiles(stack):
    out, _ = nulls.pixel_shuffle(stack, seed=8)
    valid = nulls.valid_mask(stack)
    a = sorted(map(tuple, stack[:, valid].T.tolist()))
    b = sorted(map(tuple, out[:, valid].T.tolist()))
    assert a == b and not np.array_equal(out, stack)


def test_block_shuffle_stays_in_region(stack):
    out, p = nulls.block_shuffle(stack, block=32, seed=9, region=(64, 192, 64, 256))
    outside = np.ones(stack.shape[1:], bool)
    outside[64:192, 64:256] = False
    assert np.array_equal(out[:, outside], stack[:, outside])
    assert all(2 <= dy < 6 and 2 <= dx < 8 and 2 <= sy < 6 and 2 <= sx < 8 for dy, dx, sy, sx in p["moves"])
