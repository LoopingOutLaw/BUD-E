"""Tests for env_runner: policy-in-the-loop simulation runner."""
import mujoco
import numpy as np
import torch

from bude_vla.env_runner import PolicyRolloutRunner, RolloutResult
from bude_vla.envs.so101_mjx import ARM_MODEL_PATH


class ZeroPolicy:
    @torch.no_grad()
    def sample(self, batch):
        a = np.zeros(7, dtype=np.float32)
        return torch.from_numpy(a).unsqueeze(0).unsqueeze(0)


def test_runner_produces_rollout_result_smoke():
    model = mujoco.MjModel.from_xml_path(str(ARM_MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    cube_xy = np.array([0.6, 0.0])
    runner = PolicyRolloutRunner(
        model, img_size=64, max_steps_per_try=20, max_tries=2)
    result = runner.run_one(data, ZeroPolicy(), cube_xy)
    runner.close()

    assert isinstance(result, RolloutResult)
    assert not result.success
    assert result.n_tries == 2
    assert len(result.frames) > 0
    assert all(f.shape == (64, 64, 6) for f in result.frames)
    assert all("try" in label for label in result.try_labels)


def test_runner_resets_between_tries():
    model = mujoco.MjModel.from_xml_path(str(ARM_MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    cube_xy = np.array([0.6, 0.0])
    runner = PolicyRolloutRunner(
        model, img_size=64, max_steps_per_try=10, max_tries=3)
    result = runner.run_one(data, ZeroPolicy(), cube_xy)
    runner.close()

    assert result.n_tries == 3
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    final_cube = data.xpos[cube_id, :2]
    assert np.allclose(final_cube, cube_xy, atol=0.05), (
        f"cube not reset: final={final_cube}, expected={cube_xy}")
