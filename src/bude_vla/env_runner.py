"""Policy-in-the-loop simulation runner with retry-on-failure.

Core loop: render image -> policy.sample() -> kinematic arm override -> step sim.
Cube attach/release mirrors ScriptedPickAndPlace._carry_cube_with.
On failure, reset arm to home + cube to start position, retry up to max_tries.

Exports
-------
RolloutResult       - dataclass with success, n_tries, frames, try_labels
PolicyRolloutRunner - main loop class
"""
from __future__ import annotations

import dataclasses

import mujoco
import numpy as np
import torch


TABLE_Z = 0.42
CUBE_HALF = 0.025
CARRY_ATTACH_DIST = 0.04
HOME_QPOS = np.zeros(8, dtype=np.float64)
CUBE_REST_Z = 0.445


@dataclasses.dataclass
class RolloutResult:
    success: bool
    n_tries: int
    frames: list
    try_labels: list


_PICK_INSTRUCTION = "pick up the red cube and place it in the blue target zone"
_PICK_TOKEN_IDS = None


def _pick_token_ids() -> np.ndarray:
    global _PICK_TOKEN_IDS
    if _PICK_TOKEN_IDS is None:
        from bude_vla.data.lerobot_v3 import _tokenize_instruction
        _PICK_TOKEN_IDS = _tokenize_instruction(_PICK_INSTRUCTION)
    return _PICK_TOKEN_IDS


def _ee_xyz(model, data) -> np.ndarray:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_center")
    return data.site_xpos[site_id].copy()


def _cube_xyz(model, data) -> np.ndarray:
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    return data.xpos[cube_body_id].copy()


def _target_xy(data) -> np.ndarray:
    target_body_id = mujoco.mj_name2id(
        data.model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
    return data.xpos[target_body_id, :2].copy()


def _attach_offset(model, data):
    gripper_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    gripper_xyz = data.xpos[gripper_id].copy()
    gripper_rot = data.xmat[gripper_id].reshape(3, 3).copy()
    cube_xyz = data.xpos[cube_id].copy()
    return gripper_rot.T @ (cube_xyz - gripper_xyz)


def _carry_cube_with(model, data, offset):
    if offset is None:
        return
    gripper_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    gripper_xyz = data.xpos[gripper_id].copy()
    gripper_rot = data.xmat[gripper_id].reshape(3, 3).copy()
    data.qpos[0:3] = gripper_xyz + gripper_rot @ offset


def _reset_arm_to_home(model, data):
    data.qpos[7:15] = HOME_QPOS
    data.qvel[6:15] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def _reset_cube(data, cube_xy):
    data.qpos[0:3] = [float(cube_xy[0]), float(cube_xy[1]), CUBE_REST_Z]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[0:6] = 0.0
    mujoco.mj_forward(data.model, data)


def _is_failure(model, data, step, max_steps) -> bool:
    if step >= max_steps:
        return True
    cube = data.xpos[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    ]
    if np.any(np.isnan(cube)):
        return True
    if cube[2] < TABLE_Z - 0.05 or cube[2] > 1.5:
        return True
    if np.any(np.abs(data.qpos[7:13]) > 3.5):
        return True
    return False


def _is_success(model, data, threshold: float = 0.10) -> bool:
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    target_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
    return bool(
        np.linalg.norm(data.xpos[cube_id, :2] - data.xpos[target_id, :2])
        < threshold
    )


def _build_batch(image: np.ndarray, proprio: np.ndarray,
                 text_ids: np.ndarray, domain_id: int,
                 device: str) -> dict:
    img = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1) / 255.0
    return {
        "images": img.unsqueeze(0).to(device),
        "text_ids": torch.from_numpy(text_ids).unsqueeze(0).to(device),
        "proprio": torch.from_numpy(proprio.astype(np.float32)).unsqueeze(0).to(device),
        "domain_id": torch.tensor([domain_id], dtype=torch.long).to(device),
    }


class PolicyRolloutRunner:
    def __init__(self, model, img_size: int = 224,
                 max_steps_per_try: int = 350,
                 max_tries: int = 3,
                 device: str = "cpu"):
        self.model = model
        self.img_size = img_size
        self.max_steps_per_try = max_steps_per_try
        self.max_tries = max_tries
        self.device = device
        self.renderer = mujoco.Renderer(model, height=img_size, width=img_size)
        self.text_ids = _pick_token_ids()

    def _render(self, data) -> np.ndarray:
        self.renderer.update_scene(data)
        return np.asarray(self.renderer.render()).copy()

    def run_one(self, data, policy, cube_xy) -> RolloutResult:
        frames: list = []
        try_labels: list = []
        success = False
        final_try_idx = 0

        for try_idx in range(self.max_tries):
            _reset_arm_to_home(self.model, data)
            _reset_cube(data, cube_xy)
            offset = None
            grip_close_count = 0

            for step in range(self.max_steps_per_try):
                img = self._render(data)
                proprio = data.qpos[7:15].astype(np.float32).copy()
                frames.append(img)
                try_labels.append(f"try {try_idx + 1}/{self.max_tries}")

                batch = _build_batch(img, proprio, self.text_ids,
                                     domain_id=0, device=self.device)
                actions = policy.sample(batch)
                a = actions[0, 0, :].detach().cpu().numpy()

                if np.any(np.isnan(a)):
                    arm_target = HOME_QPOS[:6].copy()
                    gripper_ctrl = 0.0
                else:
                    arm_target = np.clip(a[:6], -3.5, 3.5).astype(np.float64)
                    gripper_ctrl = float(np.clip(a[6], -1.0, 1.0))

                data.ctrl[:] = 0.0
                data.ctrl[6] = gripper_ctrl
                data.qvel[6:12] = 0.0
                data.qpos[7:13] = arm_target

                ee = _ee_xyz(self.model, data)
                cube = _cube_xyz(self.model, data)
                dist_to_cube = float(np.linalg.norm(ee - cube))

                if gripper_ctrl > 0.0 and dist_to_cube < CARRY_ATTACH_DIST:
                    grip_close_count += 1
                    if grip_close_count >= 3 and offset is None:
                        offset = _attach_offset(self.model, data)
                else:
                    grip_close_count = 0
                    if gripper_ctrl < -0.5:
                        offset = None

                _carry_cube_with(self.model, data, offset)
                mujoco.mj_step(self.model, data)
                data.qpos[7:13] = arm_target
                _carry_cube_with(self.model, data, offset)

                if _is_success(self.model, data):
                    success = True
                    frames.append(self._render(data))
                    try_labels.append(
                        f"try {try_idx + 1}/{self.max_tries} SUCCESS")
                    break

                if _is_failure(self.model, data, step, self.max_steps_per_try):
                    break

            final_try_idx = try_idx + 1
            if success:
                break

        return RolloutResult(
            success=success,
            n_tries=final_try_idx,
            frames=frames,
            try_labels=try_labels,
        )

    def run_multiple(self, data, policy, cube_positions) -> list:
        return [self.run_one(data, policy, cube_xy)
                for cube_xy in cube_positions]

    def close(self):
        self.renderer.close()
