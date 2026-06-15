import numpy as np
import pytest

from bude_vla.data.pick_recorder import record_pick_episode


@pytest.fixture(scope="module")
def episode():
    return record_pick_episode("/tmp/test_pick_recorder", episode_idx=9999)


def test_recorded_actions_are_kinematic_targets(episode):
    actions = episode["actions"]
    assert actions.ndim == 2 and actions.shape[1] == 7
    arm_actions = actions[:, :6]
    assert np.any(np.abs(arm_actions) > 1.0), (
        f"Arm joint values should exceed 1.0 (kinematic targets), "
        f"but max abs is {np.max(np.abs(arm_actions)):.4f}"
    )


def test_recorded_gripper_is_ctrl_not_target(episode):
    actions = episode["actions"]
    gripper = actions[:, 6]
    unique_vals = np.unique(np.round(gripper, 2))
    assert all(abs(v) <= 1.01 for v in unique_vals), (
        f"Gripper values should be in [-1, 1] (ctrl commands), "
        f"but got values up to {np.max(np.abs(gripper)):.4f}"
    )
