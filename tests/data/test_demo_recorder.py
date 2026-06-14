"""Tests for the v1 demo recorder."""
import numpy as np
from pathlib import Path
import tempfile

from bude_vla.envs.so101_mjx import UR5eMJMJX
from bude_vla.data.demo_recorder import collect_reach_episode, save_episode_npz


def test_collect_reach_episode_shapes():
    env = UR5eMJMJX()
    target = np.array([0.65, 0.05, 0.45], dtype=np.float32)
    ep = collect_reach_episode(env, target, n_steps=10)
    assert "images" in ep
    assert ep["images"].shape == (10, 64, 64, 3)
    assert ep["qpos"].shape[0] == 10
    assert ep["actions"].shape == (10, env.model_mj.nu), \
        f"actions shape {ep['actions'].shape} != (10, {env.model_mj.nu})"
    assert ep["instruction"] == "reach the red target"


def test_save_episode_npz():
    env = UR5eMJMJX()
    target = np.array([0.65, 0.05, 0.45], dtype=np.float32)
    ep = collect_reach_episode(env, target, n_steps=5)
    with tempfile.TemporaryDirectory() as td:
        path = save_episode_npz(ep, td)
        assert Path(path).exists()
        loaded = np.load(path, allow_pickle=True)
        assert loaded["instruction"].item() == "reach the red target"
        assert loaded["images"].shape == (5, 64, 64, 3)
        assert loaded["actions"].shape[1] == env.model_mj.nu
