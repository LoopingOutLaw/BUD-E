"""Policy-in-the-loop simulation runner with retry-on-failure.

Core loop: render image -> policy.sample() -> kinematic arm override -> step sim.
Ball carry is now handled by GraspController (grasp.py) — no kinematic
teleport. The _carry_cube_with stub is kept for compatibility but is a no-op.
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
from bude_vla.data.lerobot_v3 import _domain_from_instruction
from bude_vla.envs.so101_mjx import (
    ARM_QPOS_START, ARM_QPOS_END,
    GRIPPER_QPOS_START, GRIPPER_QPOS_END,
    CUBE_QPOS_START, CUBE_QPOS_END,
    CUBE_REST_Z,
    N_ARM_JOINTS,
    is_grasping_from_contacts,
)


# Table top is at z=0.02 in so101_mjx.py (box pos=[0,0.25,0] size=[0.35,0.10,0.02])
TABLE_Z = 0.02
HOME_QPOS = np.zeros(ARM_QPOS_END, dtype=np.float64)  # arm(5) + gripper(1) = 6


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
    # FIXED: was "ee_center" which does not exist in the MJCF; gripperframe
    # is the actual end-effector site defined in so101_new_calib.xml
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    return data.site_xpos[site_id].copy()


def _cube_xyz(model, data) -> np.ndarray:
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    return data.xpos[cube_body_id].copy()


def _target_xy(data) -> np.ndarray:
    target_body_id = mujoco.mj_name2id(
        data.model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
    return data.xpos[target_body_id, :2].copy()


def _carry_cube_with(model, data):
    """No-op: ball is carried by GraspController (grasp.py), not kinematic teleport."""
    return


def _reset_arm_to_home(model, data):
    data.qpos[ARM_QPOS_START:ARM_QPOS_END] = HOME_QPOS[:ARM_QPOS_END]
    data.qpos[GRIPPER_QPOS_START:GRIPPER_QPOS_END] = 0.3
    data.qvel[ARM_QPOS_START:GRIPPER_QPOS_END] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def _reset_cube(data, cube_xy):
    data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = [float(cube_xy[0]), float(cube_xy[1]), CUBE_REST_Z]
    data.qpos[CUBE_QPOS_START + 3:CUBE_QPOS_START + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[CUBE_QPOS_START:CUBE_QPOS_END] = 0.0
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
    if np.any(np.abs(data.qpos[ARM_QPOS_START:ARM_QPOS_END]) > 3.5):
        return True
    return False


def _is_success(model, data, threshold: float = 0.05) -> bool:
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    target_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
    return bool(
        np.linalg.norm(data.xpos[cube_id, :2] - data.xpos[target_id, :2])
        < threshold
    )


def _build_proprio(model, data, state_dim: int) -> np.ndarray:
    """Build proprio vector matching training dimension."""
    gripperframe_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    target_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")

    base = data.qpos[ARM_QPOS_START:GRIPPER_QPOS_END].astype(np.float32).copy()

    if state_dim == 6:
        return base
    elif state_dim == 7:
        is_g = is_grasping_from_contacts(model, data)
        return np.concatenate([base, [is_g]]).astype(np.float32)
    elif state_dim == 9:
        gripper_pos = data.site_xpos[gripperframe_id]
        target_pos = data.xpos[target_body_id]
        target_rel = target_pos[:2] - gripper_pos[:2]
        is_g = is_grasping_from_contacts(model, data)
        return np.concatenate([base, target_rel, [is_g]]).astype(np.float32)
    else:
        raise ValueError(f"Unsupported state_dim: {state_dim}")


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
                 n_history_frames: int = 1,
                 state_dim: int = 6,
                 ensembling: bool = False,
                 ensembling_k: float = 0.5,
                 arm_smooth_steps: int = 1,
                 arm_step_frac: float = 1.0):
        self.model = model
        self.img_size = img_size
        self.max_steps_per_try = max_steps_per_try
        self.max_tries = max_tries
        self.device = device
        self.n_history_frames = max(1, int(n_history_frames))
        self.state_dim = state_dim
        self._frame_buffer: list = []
        self.renderer = mujoco.Renderer(model, height=img_size, width=img_size)
        self.overhead_cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "front_top")
        self.wrist_cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
        self.portfolio_cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "portfolio")
        self.text_ids = _pick_token_ids()
        self.ensembling = ensembling
        self.ensembling_k = ensembling_k  # weight of new chunk vs in-queue action
        self._action_queue: list = []  # pending denormalized actions
        self.arm_smooth_steps = max(1, int(arm_smooth_steps))
        self.arm_step_frac = float(arm_step_frac)

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

    def _render(self, data, camera: str = "default") -> np.ndarray:
        if camera == "portfolio":
            self.renderer.update_scene(data, camera=self.portfolio_cam_id)
            return np.asarray(self.renderer.render()).copy()
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
                viewer=None, step_delay: float = 0.0,
                record_video_mode: bool = False,
                record_camera: str = "default") -> RolloutResult:
        frames: list = []
        try_labels: list = []
        success = False
        final_try_idx = 0

        ARM_SMOOTH_STEPS = max(6, self.arm_smooth_steps) if record_video_mode else self.arm_smooth_steps
        ARM_STEP_FRAC = (1.0 / ARM_SMOOTH_STEPS) if record_video_mode else self.arm_step_frac

        def _smooth_arm_to(target_qpos, try_idx_inner):
            cur = data.qpos[ARM_QPOS_START:ARM_QPOS_END].astype(np.float64).copy()
            tgt = np.clip(target_qpos, -3.5, 3.5).astype(np.float64)
            for k in range(ARM_SMOOTH_STEPS):
                err = tgt - cur
                cur = cur + err * ARM_STEP_FRAC
                data.ctrl[:] = 0.0
                data.ctrl[GRIPPER_QPOS_START] = gripper_ctrl
                data.qvel[ARM_QPOS_START:ARM_QPOS_END] = 0.0
                data.qpos[ARM_QPOS_START:ARM_QPOS_END] = cur
                _carry_cube_with(self.model, data)
                mujoco.mj_step(self.model, data)
                if record_video_mode and record_camera != "default":
                    img_obs = self._render(data, camera="default")
                    self._stacked_view(img_obs)
                    img_vid = self._render(data, camera=record_camera)
                    frames.append(img_vid)
                else:
                    img_mid = self._render(data, camera=record_camera)
                    stacked_mid = self._stacked_view(img_mid)
                    frames.append(stacked_mid)
                try_labels.append(
                    f"try {try_idx_inner + 1}/{self.max_tries}")
            data.qpos[ARM_QPOS_START:ARM_QPOS_END] = tgt
            _carry_cube_with(self.model, data)
            return tgt

        for try_idx in range(self.max_tries):
            _reset_arm_to_home(self.model, data)
            _reset_cube(data, cube_xy)
            gripper_ctrl = 0.0
            chunk = None
            cursor = 0
            self._frame_buffer = []
            self._action_queue = []

            for step in range(self.max_steps_per_try):
                img = self._render(data)
                stacked = self._stacked_view(img)
                arm_proprio = _build_proprio(self.model, data, self.state_dim)
                if record_video_mode and record_camera != "default":
                    frames.append(self._render(data, camera=record_camera))
                else:
                    frames.append(stacked)
                try_labels.append(f"try {try_idx + 1}/{self.max_tries}")

                if self.ensembling:
                    if not self._action_queue:
                        batch = _build_batch(stacked, arm_proprio, self.text_ids,
                                             _PICK_INSTRUCTION,
                                             domain_id=_domain_from_instruction(_PICK_INSTRUCTION),
                                             device=self.device)
                        new_chunk = policy.sample(batch)[0].detach().cpu().numpy()
                        if self._use_norm:
                            new_chunk = denormalize_actions(
                                new_chunk, self._action_lo, self._action_hi)
                        q = list(self._action_queue)
                        for i, new_a in enumerate(new_chunk):
                            if i < len(q):
                                q[i] = (self.ensembling_k * q[i]
                                        + (1 - self.ensembling_k) * new_a)
                            else:
                                q.append(new_a)
                        self._action_queue = q
                    a = self._action_queue.pop(0)
                else:
                    if chunk is None or cursor >= chunk.shape[0]:
                        batch = _build_batch(stacked, arm_proprio, self.text_ids,
                                             _PICK_INSTRUCTION,
                                             domain_id=_domain_from_instruction(_PICK_INSTRUCTION),
                                             device=self.device)
                        chunk = policy.sample(batch)[0].detach().cpu().numpy()
                        cursor = 0
                    a = chunk[cursor]
                    cursor += 1
                    if self._use_norm:
                        a = denormalize_actions(a, self._action_lo, self._action_hi)

                if np.any(np.isnan(a)):
                    arm_target = HOME_QPOS[ARM_QPOS_START:ARM_QPOS_END].copy()
                    gripper_ctrl = 0.0
                else:
                    arm_target = np.clip(a[:N_ARM_JOINTS], -3.5, 3.5).astype(np.float64)
                    gripper_ctrl = float(np.clip(a[N_ARM_JOINTS], -1.5, 1.5))

                _carry_cube_with(self.model, data)
                _smooth_arm_to(arm_target, try_idx)

                if _is_success(self.model, data):
                    success = True
                    if record_video_mode and record_camera != "default":
                        frames.append(self._render(data, camera=record_camera))
                    else:
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
