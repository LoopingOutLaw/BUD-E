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
from pathlib import Path

import mujoco
import numpy as np
import torch

from bude_vla.data.action_normalization import (
    DEFAULT_HI,
    DEFAULT_LO,
    denormalize_actions,
    load_action_stats,
)


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
                 text_ids: np.ndarray, instruction: str, domain_id: int,
                 device: str) -> dict:
    img = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1) / 255.0
    return {
        "images": img.unsqueeze(0).to(device),
        "text_ids": torch.from_numpy(text_ids).unsqueeze(0).to(device),
        "instruction": [instruction],
        "proprio": torch.from_numpy(proprio.astype(np.float32)).unsqueeze(0).to(device),
        "domain_id": torch.tensor([domain_id], dtype=torch.long).to(device),
    }


class PolicyRolloutRunner:
    def __init__(self, model, img_size: int = 224,
                 max_steps_per_try: int = 350,
                 max_tries: int = 3,
                 device: str = "cpu",
                 action_norm_root: str | None = None,
                 action_lo: np.ndarray | list | None = None,
                 action_hi: np.ndarray | list | None = None,
                 n_history_frames: int = 1):
        self.model = model
        self.img_size = img_size
        self.max_steps_per_try = max_steps_per_try
        self.max_tries = max_tries
        self.device = device
        self.n_history_frames = max(1, int(n_history_frames))
        self._frame_buffer: list = []
        self.renderer = mujoco.Renderer(model, height=img_size, width=img_size)
        self.overhead_cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "front_top")
        self.wrist_cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "gripper_cam")
        self.text_ids = _pick_token_ids()

        # ── action normalization ──────────────────────────────────────────
        # Priority: explicit lo/hi > data-root file > DEFAULT
        # ALWAYS denormalize because training normalizes to [-1,1].
        if action_lo is not None and action_hi is not None:
            self._action_lo = np.asarray(action_lo, dtype=np.float32)
            self._action_hi = np.asarray(action_hi, dtype=np.float32)
        elif action_norm_root is not None:
            self._action_lo, self._action_hi = load_action_stats(
                Path(action_norm_root) / "meta" / "info.json"
            )
        else:
            self._action_lo = DEFAULT_LO.copy()
            self._action_hi = DEFAULT_HI.copy()
        self._use_norm = True

    def _render(self, data) -> np.ndarray:
        self.renderer.update_scene(data, camera=self.overhead_cam_id)
        img_overhead = np.asarray(self.renderer.render()).copy()
        self.renderer.update_scene(data, camera=self.wrist_cam_id)
        img_wrist = np.asarray(self.renderer.render()).copy()
        return np.concatenate([img_overhead, img_wrist], axis=-1)

    def _stacked_view(self, frame: np.ndarray) -> np.ndarray:
        if self.n_history_frames <= 1:
            return frame
        self._frame_buffer.append(frame)
        if len(self._frame_buffer) > self.n_history_frames:
            self._frame_buffer.pop(0)
        while len(self._frame_buffer) < self.n_history_frames:
            self._frame_buffer.insert(0, frame)
        return np.concatenate(self._frame_buffer, axis=-1)

    def run_one(self, data, policy, cube_xy,
                viewer=None, step_delay: float = 0.0) -> RolloutResult:
        frames: list = []
        try_labels: list = []
        success = False
        final_try_idx = 0

        ARM_SMOOTH_STEPS = 14
        ARM_STEP_FRAC = 0.22

        def _smooth_arm_to(target_qpos):
            cur = data.qpos[7:13].astype(np.float64).copy()
            tgt = np.clip(target_qpos, -3.5, 3.5).astype(np.float64)
            for k in range(ARM_SMOOTH_STEPS):
                err = tgt - cur
                cur = cur + err * ARM_STEP_FRAC
                data.ctrl[:] = 0.0
                data.ctrl[6] = gripper_ctrl
                data.qvel[6:12] = 0.0
                data.qpos[7:13] = cur
                _carry_cube_with(self.model, data, offset)
                mujoco.mj_step(self.model, data)
                img_mid = self._render(data)
                stacked_mid = self._stacked_view(img_mid)
                frames.append(stacked_mid)
                try_labels.append(
                    f"try {try_idx + 1}/{self.max_tries}")
            data.qpos[7:13] = tgt
            _carry_cube_with(self.model, data, offset)
            return tgt

        for try_idx in range(self.max_tries):
            _reset_arm_to_home(self.model, data)
            _reset_cube(data, cube_xy)
            offset = None
            grip_close_count = 0
            gripper_ctrl = 0.0
            chunk = None
            cursor = 0
            self._frame_buffer = []

            for step in range(self.max_steps_per_try):
                img = self._render(data)
                stacked = self._stacked_view(img)
                arm_proprio = data.qpos[7:15].astype(np.float32).copy()
                frames.append(stacked)
                try_labels.append(f"try {try_idx + 1}/{self.max_tries}")

                if chunk is None or cursor >= chunk.shape[0]:
                    batch = _build_batch(stacked, arm_proprio, self.text_ids,
                                         _PICK_INSTRUCTION, domain_id=0,
                                         device=self.device)
                    chunk = policy.sample(batch)[0].detach().cpu().numpy()
                    cursor = 0

                a = chunk[cursor]
                cursor += 1

                if self._use_norm:
                    a = denormalize_actions(a, self._action_lo, self._action_hi)

                if np.any(np.isnan(a)):
                    arm_target = HOME_QPOS[:6].copy()
                    gripper_ctrl = 0.0
                else:
                    arm_target = np.clip(a[:6], -3.5, 3.5).astype(np.float64)
                    gripper_ctrl = float(np.clip(a[6], -1.0, 1.0))

                _carry_cube_with(self.model, data, offset)
                _smooth_arm_to(arm_target)

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

                if _is_success(self.model, data):
                    success = True
                    frames.append(self._stacked_view(self._render(data)))
                    try_labels.append(
                        f"try {try_idx + 1}/{self.max_tries} SUCCESS")
                    break

                if _is_failure(self.model, data, step, self.max_steps_per_try):
                    break

                if viewer is not None:
                    viewer.sync()
                if step_delay > 0:
                    import time as _t
                    _t.sleep(step_delay)

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
