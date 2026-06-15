"""Show that demo recorder emits 8-dim arm+gripper proprio (not the full
15-dim qpos which includes the cube's freejoint).

8 dims = 6 arm hinge joints + 2 finger slide joints (coupled but both observed).
"""
import numpy as np
from bude_vla.envs.so101_mjx import UR5eMJMJX
from bude_vla.data.demo_recorder import collect_reach_episode, collect_push_episode


def test_reach_episode_proprio_is_8_dim():
    env = UR5eMJMJX()
    ep = collect_reach_episode(env, np.array([0.6, 0.0, 0.55], dtype=np.float32), n_steps=5)
    assert "proprio" in ep, "demo_recorder should emit 'proprio' for the model"
    assert ep["proprio"].shape == (5, 8), f"want (T, 8) got {ep['proprio'].shape}"


def test_push_episode_proprio_is_8_dim():
    env = UR5eMJMJX()
    ep = collect_push_episode(env, np.array([0.25, 0.0], dtype=np.float32), n_steps=5)
    assert "proprio" in ep
    assert ep["proprio"].shape == (5, 8), f"want (T, 8) got {ep['proprio'].shape}"
