"""Integration test: proves --augment sets the BUDETrainingDataset flag and
that augmented images actually differ from non-augmented ones per call."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from bude_vla.data.lerobot_v3 import BUDETrainingDataset

DATA_ROOT = Path("/home/aditya/bude_vla/data/pick_v3_224_prop11")


def test_dataset_augment_flag_propagates():
    """Same root, augment=True produces different images than augment=False."""
    if not DATA_ROOT.exists():
        import pytest
        pytest.skip(f"dataset not at {DATA_ROOT}")

    ds_off = BUDETrainingDataset(DATA_ROOT, chunk_size=4, augment=False)
    ds_off.read()
    img_off = ds_off[42]["images"]

    ds_on = BUDETrainingDataset(DATA_ROOT, chunk_size=4, augment=True)
    ds_on.read()
    samples = [ds_on[42]["images"] for _ in range(8)]

    # All comparisons run on the same window, same dataset, same idx.
    # aug=True -> 8 random crops / brightnesses should mostly differ from off
    assert img_off.shape == samples[0].shape == (3, 224, 224)
    differs = sum(1 for s in samples if not torch.allclose(img_off, s, atol=1e-6))
    # Brightness-only moves every pixel; crop_pad=0 (default) -> identical
    # when crop is disabled. With default crop_pad=8, most should differ.
    # Use a relaxed threshold: at least one different sample proves flag works.
    assert differs >= 1, (
        "All augmented samples were identical to non-augmented; flag "
        "passed but pipeline didn't actually randomize."
    )


def test_dataset_two_fetched_augmented_images_vary():
    """Two consecutive fetches with augment=True must not be byte-identical
    (random crop draws fresh translate on every __getitem__)."""
    if not DATA_ROOT.exists():
        import pytest
        pytest.skip(f"dataset not at {DATA_ROOT}")

    ds = BUDETrainingDataset(DATA_ROOT, chunk_size=4, augment=True)
    ds.read()
    a = ds[0]["images"]
    b = ds[0]["images"]
    assert not torch.allclose(a, b, atol=1e-6), (
        "Two fetches of the same idx returned identical augmented images — "
        "_rng or augment not active."
    )
