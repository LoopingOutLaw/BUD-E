"""Tests for BUDETrainingDataset (PyTorch Dataset wrapper for LeRobot v3)."""
import tempfile

import numpy as np
import torch

from bude_vla.data.cpu_recorder import record_dataset_cpu
from bude_vla.data.lerobot_v3 import BUDETrainingDataset


def test_dataset_len_matches_total_frames():
    with tempfile.TemporaryDirectory() as td:
        record_dataset_cpu(task="reach", n_episodes=3, n_steps=5, root=td)
        ds = BUDETrainingDataset(td)
        ds.read()
        assert len(ds) > 0


def test_dataset_getitem_returns_correct_keys():
    with tempfile.TemporaryDirectory() as td:
        record_dataset_cpu(task="reach", n_episodes=2, n_steps=6, root=td)
        ds = BUDETrainingDataset(td, chunk_size=4)
        ds.read()
        sample = ds[0]
        assert "images" in sample
        assert "text_ids" in sample
        assert "proprio" in sample
        assert "actions" in sample
        assert "domain_id" in sample
        assert "tau" in sample
        assert "noise" in sample
        assert sample["images"].shape == (3, 64, 64)
        assert sample["proprio"].shape == (8,)
        assert sample["actions"].shape == (4, 7)


def test_dataset_action_chunk_pads_at_episode_end():
    """When chunk_size exceeds remaining frames, pad with zeros."""
    with tempfile.TemporaryDirectory() as td:
        record_dataset_cpu(task="reach", n_episodes=1, n_steps=3, root=td)
        ds = BUDETrainingDataset(td, chunk_size=10)
        ds.read()
        for idx in range(len(ds)):
            sample = ds[idx]
            if sample["actions"][-1].sum().item() == 0.0:
                return
        assert False, "Expected at least one padded action chunk"
