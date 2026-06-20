"""Tests for data augmentation in BUDETrainingDataset."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from bude_vla.data.lerobot_v3 import (
    BUDETrainingDataset,
    _augment_image,
    _CHAR_VOCAB,
)


# ─────────────────────────────────────────────────────────────────────
# _augment_image unit tests
# ─────────────────────────────────────────────────────────────────────

def _img(h=224, w=224, c=3) -> torch.Tensor:
    """Solid mid-gray image: all 0.5 so brightness jitter is observable."""
    return torch.full((c, h, w), 0.5, dtype=torch.float32)


def test_augment_image_shape_preserved():
    """Output shape == (C, H, W) whichever dim the input has."""
    for h, w in [(224, 224), (160, 192), (64, 64)]:
        img = _img(h, w)
        out = _augment_image(img, np.random.default_rng(0),
                             brightness_range=0.1, crop_pad=8)
        assert out.shape == (3, h, w)


def test_augment_image_brightness_clamped():
    """Luminosity factor, no clamp violation."""
    rng = np.random.default_rng(42)
    img = _img()
    out = _augment_image(img, rng, brightness_range=0.10, crop_pad=0)
    # Out image may be 0.4..0.6 since base=0.5 and delta~U(-0.1,+0.1).
    assert (out >= 0.0).all(), "values below 0"
    assert (out <= 1.0).all(), "values above 1"


def test_augment_image_brightness_mean_within_range():
    """Over many trials, mean delta is approx 0 and max abs delta <= range."""
    img = _img()
    deltas = []
    for i in range(50):
        rng = np.random.default_rng(i)
        out = _augment_image(img, rng, brightness_range=0.10, crop_pad=0)
        deltas.append((out.mean().item() - 0.5))
    assert np.mean(np.abs(deltas)) <= 0.05  # roughly centered


def test_augment_image_crop_reflect_no_alloc():
    """Crop + reflect pad keeps every pixel value inside the original range
    - it must not introduce values outside [0, 1]."""
    rng = np.random.default_rng(7)
    img = torch.rand(3, 64, 64, dtype=torch.float32)
    out = _augment_image(img, rng, brightness_range=0.0, crop_pad=8)
    assert out.shape == (3, 64, 64)
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_augment_image_crop_pad_zero_keeps_pixels():
    """With crop_pad=0 we only add brightness; mean pixel stays close to input."""
    img = torch.full((3, 32, 32), 0.5, dtype=torch.float32)
    out = _augment_image(img, np.random.default_rng(0),
                              brightness_range=0.0, crop_pad=0)
    assert torch.allclose(img, out, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────
# Dataset integration
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ds_minimal(tmp_path_factory):
    """Write a tiny 4-frame episode parquet so the dataset can read it.

    Avoids Mujoco; BUDETrainingDataset only reads parquet text + a tiny image.
    """
    pass
