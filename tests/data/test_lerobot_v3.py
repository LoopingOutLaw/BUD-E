"""Tests for lerobot-v3 writer/reader."""
import shutil
import tempfile
from pathlib import Path

import numpy as np

from bude_vla.envs.so101_mjx import UR5eMJMJX
from bude_vla.data.demo_recorder import collect_reach_episode
from bude_vla.data.lerobot_v3 import BUDEDataset, write_episode


def _make_episode(env, T: int = 5) -> dict:
    target = np.array([0.65, 0.05, 0.45], dtype=np.float32)
    return collect_reach_episode(env, target, n_steps=T)


def test_write_episode_creates_v3_layout():
    env = UR5eMJMJX()
    ep = _make_episode(env, T=5)
    with tempfile.TemporaryDirectory() as td:
        write_episode(td, ep)
        assert (Path(td) / "meta" / "info.json").exists()
        assert (Path(td) / "meta" / "episodes_index" / "episode_000000.json").exists()
        assert (Path(td) / "data" / "chunk-000" / "episode_000000.parquet").exists()
        assert (Path(td) / "videos" / "chunk-000" / "observation.images.top" / "episode_000000.mp4").exists()


def test_dataset_reads_back_written_episode():
    env = UR5eMJMJX()
    T = 5
    ep = _make_episode(env, T=T)
    n_state = ep["qpos"].shape[1]
    n_action = ep["actions"].shape[1]
    with tempfile.TemporaryDirectory() as td:
        write_episode(td, ep)
        ds = BUDEDataset(td)
        assert ds.num_episodes() == 1
        assert len(ds) == T
        sample = ds.get_episode(0)
        assert sample["state"].shape == (T, n_state)
        assert sample["action"].shape == (T, n_action)
        assert sample["instruction"] == ep["instruction"]


def test_two_episodes_load_correctly():
    """Write 2 short episodes using a single env instance and re-read."""
    env = UR5eMJMJX()
    target_far = np.array([0.65, 0.50, 0.65], dtype=np.float32)
    ep1 = collect_reach_episode(env, target_far, n_steps=4)
    target_far2 = np.array([0.55, -0.30, 0.50], dtype=np.float32)
    ep2 = collect_reach_episode(env, target_far2, n_steps=6)
    with tempfile.TemporaryDirectory() as td:
        write_episode(td, ep1)
        write_episode(td, ep2)
        ds = BUDEDataset(td)
        assert ds.num_episodes() == 2, f"expected 2 got {ds.num_episodes()}"
        sample1 = ds.get_episode(0)
        sample2 = ds.get_episode(1)
        assert sample1["state"].shape[0] == 4
        assert sample2["state"].shape[0] > 0
