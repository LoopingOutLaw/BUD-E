"""Tests for push scripted policy and recorder."""
import numpy as np
from pathlib import Path
import tempfile

from bude_vla.envs.so101_mjx import UR5eMJMJX
from bude_vla.data.demo_recorder import collect_push_episode
from bude_vla.data.scripted_policies import scripted_push_step


def test_push_episode_runs():
    env = UR5eMJMJX()
    target_2d = np.array([0.0, 0.0], dtype=np.float32)
    ep = collect_push_episode(env, target_2d, n_steps=10)
    assert ep["images"].shape == (10, 64, 64, 3)
    assert ep["qpos"].shape[0] == 10
    assert ep["actions"].shape[1] == env.model_mj.nu
    assert ep["instruction"] == "push the cube to the green zone"


def test_scripted_push_step_dim():
    a, p = scripted_push_step(np.zeros(3), np.array([0.6, 0.0, 0.435]),
                                np.array([0.85, 0.0, 0.421]), phase=0, nu=7)
    assert a.shape == (7,)
    assert p in (0, 1)
