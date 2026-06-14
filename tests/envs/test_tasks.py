"""Tests for task wrappers."""
import numpy as np
from bude_vla.envs.so101_mjx import UR5eMJMJX
from bude_vla.envs.tasks.reach import ReachTask
from bude_vla.envs.tasks.push_pick import PushTask, PickPlaceTask


def test_reach_task_resets_target():
    env = UR5eMJMJX()
    task = ReachTask(env)
    rng = np.random.default_rng(42)
    task.reset_target(rng)
    assert task.target_pos.shape == (3,)
    assert task.target_pos[2] > 0.4   # above table


def test_reach_task_distance_to_target():
    env = UR5eMJMJX()
    task = ReachTask(env)
    rng = np.random.default_rng(0)
    task.reset_target(rng)
    # distance is positive
    s = env.reset()
    assert task.distance_to_target(s) >= 0.0


def test_push_task_reset_repositions_cube_and_target():
    env = UR5eMJMJX()
    task = PushTask(env)
    rng = np.random.default_rng(0)
    task.reset_episode(rng)
    assert task.cube_pos.shape == (3,)
    assert task.target_pos.shape == (3,)


def test_pickplace_resets_correctly():
    env = UR5eMJMJX()
    task = PickPlaceTask(env)
    rng = np.random.default_rng(0)
    task.reset_episode(rng)
    assert task.cube_pos[2] > 0.4
    assert task.cube_in_gripper is False
    assert task.success is False
