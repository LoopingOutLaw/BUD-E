"""Action-space helpers for policy rollouts and dataset conversion."""
from __future__ import annotations

import numpy as np
import mujoco

from bude_vla.envs.so101_mjx import GRIPPER_QPOS_START, N_ARM_JOINTS
from bude_vla.ik import IKController

WRIST_FLEX_LOCK = np.pi / 2
WRIST_ROLL_LOCK = np.pi / 2


def action_space_from_cfg(cfg) -> str:
    return str(getattr(cfg, "action_space", "joint_abs") or "joint_abs")


def ee_delta_scale_from_cfg(cfg) -> float:
    return float(getattr(cfg, "ee_delta_scale", 0.05) or 0.05)


def end_effector_position_for_qpos(model: mujoco.MjModel, data: mujoco.MjData,
                                   arm_qpos: np.ndarray, gripper_qpos: float = 0.3,
                                   site_name: str = "gripperframe") -> np.ndarray:
    """Return TCP position for a candidate SO-101 arm pose."""
    data.qpos[:N_ARM_JOINTS] = np.asarray(arm_qpos, dtype=np.float64)[:N_ARM_JOINTS]
    data.qpos[GRIPPER_QPOS_START] = float(gripper_qpos)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Site {site_name!r} not found")
    return data.site_xpos[site_id].copy()


def joint_action_to_ee_delta(model: mujoco.MjModel, fk_data: mujoco.MjData,
                             state: np.ndarray, action: np.ndarray,
                             max_delta: float = 0.08) -> np.ndarray:
    """Convert old [5 joint targets + gripper] action into [TCP delta xyz + gripper]."""
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    cur_q = state[:N_ARM_JOINTS]
    target_q = action[:N_ARM_JOINTS]
    gripper = float(action[N_ARM_JOINTS])
    cur_pos = end_effector_position_for_qpos(model, fk_data, cur_q, state[GRIPPER_QPOS_START])
    target_pos = end_effector_position_for_qpos(model, fk_data, target_q, gripper)
    delta = np.clip(target_pos - cur_pos, -float(max_delta), float(max_delta))
    return np.asarray([delta[0], delta[1], delta[2], gripper], dtype=np.float32)


def make_ik_controller(model: mujoco.MjModel, data: mujoco.MjData) -> IKController:
    return IKController(model, data, end_effector_site="gripperframe", damping=0.08, max_dq=0.35)


def apply_policy_action(model: mujoco.MjModel, data: mujoco.MjData, action: np.ndarray,
                        cfg, ik: IKController | None = None,
                        contact_close_reflex: bool = False,
                        close_active: bool = False,
                        contact_close_value: float = -1.0) -> tuple[np.ndarray, float]:
    """Apply a policy action to MuJoCo controls and return arm_target, gripper_ctrl."""
    action = np.asarray(action, dtype=np.float64)
    gripper_ctrl = float(np.clip(action[-1], -1.5, 1.5))
    if contact_close_reflex and close_active:
        gripper_ctrl = min(gripper_ctrl, contact_close_value)

    if action_space_from_cfg(cfg) == "ee_delta":
        if ik is None:
            raise ValueError("ee_delta action execution requires an IKController")
        delta = np.clip(action[:3], -ee_delta_scale_from_cfg(cfg), ee_delta_scale_from_cfg(cfg))
        target_pos = ik.get_ee_position() + delta
        ctrl = ik.step_toward_target(
            target_pos,
            gripper_action=0.0,
            gain=1.0,
            locked_joints=[3, 4],
        )
        arm_target = np.clip(ctrl[:N_ARM_JOINTS], -3.5, 3.5).astype(np.float64)
        arm_target[3] = WRIST_FLEX_LOCK
        arm_target[4] = WRIST_ROLL_LOCK
    else:
        arm_target = np.clip(action[:N_ARM_JOINTS], -3.5, 3.5).astype(np.float64)

    data.ctrl[:N_ARM_JOINTS] = arm_target
    data.ctrl[GRIPPER_QPOS_START] = gripper_ctrl
    return arm_target, gripper_ctrl
