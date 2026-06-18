"""Round-trip tests for action normalization."""
import numpy as np

from bude_vla.data.action_normalization import (
    DEFAULT_HI,
    DEFAULT_LO,
    compute_action_stats,
    denormalize_actions,
    normalize_actions,
    pad_scale,
)


def test_normalize_denormalize_roundtrip_strict():
    rng = np.random.default_rng(42)
    lo = np.array([-1.0, 0.0, -3.0, -1.5, 2.0, -0.5, -1.0], dtype=np.float32)
    hi = np.array([2.0, 1.0, 3.5, 1.5, 4.0, 0.4, 1.0], dtype=np.float32)
    arr = rng.uniform(lo, hi, size=(64, 7)).astype(np.float32)
    norm = normalize_actions(arr, lo, hi)
    assert norm.min() >= -1.0 and norm.max() <= 1.0, "values should land in [-1,1]"
    rec = denormalize_actions(norm, lo, hi)
    assert np.allclose(rec, arr, atol=1e-5), "round-trip within 1e-5"


def test_normalize_denormalize_edge_values():
    lo = DEFAULT_LO.copy()
    hi = DEFAULT_HI.copy()
    for val in [lo, hi, np.zeros_like(lo), (lo + hi) / 2.0]:
        arr = val.reshape(1, -1).astype(np.float32)
        rec = denormalize_actions(normalize_actions(arr, lo, hi), lo, hi)
        assert np.allclose(rec, arr, atol=1e-5)


def test_normalize_handles_chunked_input():
    lo = np.array([-3.0] * 7, dtype=np.float32)
    hi = np.array([3.0] * 7, dtype=np.float32)
    arr = np.random.default_rng(0).uniform(lo, hi, size=(5, 4, 7)).astype(np.float32)
    rec = denormalize_actions(normalize_actions(arr, lo, hi), lo, hi)
    assert np.allclose(rec, arr, atol=1e-5)


def test_compute_action_stats_basic():
    rng = np.random.default_rng(7)
    eps = [rng.uniform(-1, 1, (10, 3)).astype(np.float32) for _ in range(3)]
    lo, hi = compute_action_stats(eps)
    assert lo.shape == (3,) and hi.shape == (3,)
    for lp, hp in zip(lo, hi):
        assert hp >= lp


def test_pad_scale_widens_range():
    lo, hi = pad_scale(0.0, 1.0, margin=0.02)
    assert lo < 0.0 and hi > 1.0
    assert abs(lo - -0.02) < 1e-5
    assert abs(hi - 1.02) < 1e-5


def test_pad_scale_handles_zero_span():
    lo, hi = pad_scale(1.0, 1.0)
    assert hi > lo
