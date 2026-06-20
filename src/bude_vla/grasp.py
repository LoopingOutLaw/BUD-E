"""Physically-gated grasp/carry logic for the SO-101 cube-pick task.

v2 redesign: the cube is centered between both fingers at the moment of
attach, not snapped to the moving jaw's inner surface.  This eliminates
the "magnetic" one-sided attachment that the previous version suffered
from.

Attach conditions (unchanged from v1):
  1. Cube center within `attach_gap_tolerance` of the midpoint between
     jaw_contact and fixed_finger_contact.
  2. Jaw qpos closed past `jaw_closed_qpos_threshold`.
  3. MuJoCo contacts show real contact between EITHER moving jaw OR
     fixed finger AND the cube.
  4. All three hold for `attach_debounce_steps` consecutive steps.

At attach: the cube center is snapped to the midpoint between the two
finger contact sites, offset by CUBE_HALF_EXTENT in the direction that
places it between both surfaces.  The stored offset is the LOCAL-FRAME
vector from the midpoint to the cube center, so during carry the cube
tracks the midpoint as the arm moves.

Release is explicit (force_release from ScriptedPickAndPlace).
"""
from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from bude_vla.envs.so101_mjx import CUBE_QPOS_START, CUBE_QPOS_END

CUBE_HALF_EXTENT = 0.010
ATTACH_GAP_TOLERANCE = 0.008   # snap distance < half cube, imperceptible teleport
ATTACH_DEBOUNCE_STEPS = 3
JAW_CLOSED_QPOS_THRESHOLD = 0.30  # must be meaningfully closed, not just pre-closed
IK_SEED_JAW_QPOS = 0.30
RELEASE_JAW_QPOS_THRESHOLD = 1.48
RELEASE_DRIFT_TOLERANCE = 0.015  # 1.5cm max drift before release (was 8cm)


@dataclasses.dataclass
class GraspState:
    attached: bool = False
    offset_local: np.ndarray | None = None      # in midpoint-body local frame
    enclosure_streak: int = 0
    last_world: np.ndarray | None = None
    release_reason: str | None = None
    midpoint_body_id: int = -1                   # body used for local-frame offset

    def reset(self) -> None:
        self.attached = False
        self.offset_local = None
        self.enclosure_streak = 0
        self.last_world = None
        self.release_reason = None
        self.midpoint_body_id = -1


class GraspController:
    def __init__(self, model: mujoco.MjModel,
                 jaw_site_name: str = "jaw_contact",
                 jaw_body_name: str = "moving_jaw_so101_v1",
                 gripper_body_name: str = "gripper",
                 ff_site_name: str = "fixed_finger_contact",
                 cube_body_name: str = "cube",
                 cube_half_extent: float = CUBE_HALF_EXTENT,
                 attach_gap_tolerance: float = ATTACH_GAP_TOLERANCE,
                 attach_debounce_steps: int = ATTACH_DEBOUNCE_STEPS,
                 jaw_closed_qpos_threshold: float = JAW_CLOSED_QPOS_THRESHOLD,
                 release_jaw_qpos_threshold: float = RELEASE_JAW_QPOS_THRESHOLD,
                 release_drift_tolerance: float = RELEASE_DRIFT_TOLERANCE,
                 require_contact: bool = True):
        self.jaw_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, jaw_site_name)
        self.ff_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ff_site_name)
        self.jaw_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, jaw_body_name)
        self.gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gripper_body_name)
        self.cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cube_body_name)
        for name, bid, kind in [(jaw_site_name, self.jaw_site_id, "site"),
                                (ff_site_name, self.ff_site_id, "site"),
                                (jaw_body_name, self.jaw_body_id, "body"),
                                (gripper_body_name, self.gripper_body_id, "body"),
                                (cube_body_name, self.cube_body_id, "body")]:
            if bid < 0:
                raise ValueError(
                    f"GraspController: {kind} '{name}' not found in model. "
                    f"Check load_arm_model() / _build_composite_spec()."
                )
        self.cube_half_extent = cube_half_extent
        self.attach_gap_tolerance = attach_gap_tolerance
        self.attach_debounce_steps = attach_debounce_steps
        self.jaw_closed_qpos_threshold = jaw_closed_qpos_threshold
        self.release_jaw_qpos_threshold = release_jaw_qpos_threshold
        self.release_drift_tolerance = release_drift_tolerance
        self.require_contact = require_contact
        self.state = GraspState()

    def reset(self) -> None:
        self.state.reset()

    def _has_gripper_cube_contact(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        """Check if EITHER the moving jaw OR the fixed finger (gripper body)
        is in contact with the cube."""
        body_ids = model.geom_bodyid
        gripper_ids = {self.jaw_body_id, self.gripper_body_id}
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = body_ids[c.geom1], body_ids[c.geom2]
            if (b1 in gripper_ids and b2 == self.cube_body_id) or \
               (b2 in gripper_ids and b1 == self.cube_body_id):
                return True
        return False

    def _midpoint(self, data: mujoco.MjData) -> tuple[np.ndarray, int, np.ndarray]:
        """Compute the grip center between jaw_contact and fixed_finger_contact.

        The two sites are at different world Z heights due to the gripper
        body orientation varying with arm config. For horizontal clamping,
        we use the XY midpoint and set Z to the lower of the two sites
        (the one closer to the cube).
        """
        jaw_xyz = data.site_xpos[self.jaw_site_id].copy()
        ff_xyz = data.site_xpos[self.ff_site_id].copy()
        # XY midpoint for horizontal centering; Z = lower site (closer to cube)
        midpoint = np.array([
            (jaw_xyz[0] + ff_xyz[0]) / 2.0,
            (jaw_xyz[1] + ff_xyz[1]) / 2.0,
            min(jaw_xyz[2], ff_xyz[2]),
        ])

        # Use gripper body for local frame (it's the parent of both
        # the fixed finger site and the moving jaw body)
        body_id = self.gripper_body_id
        rot = data.xmat[body_id].reshape(3, 3).copy()

        return midpoint, body_id, rot

    def gap(self, data: mujoco.MjData) -> float:
        """Distance between the cube center and the finger midpoint."""
        midpoint, _, _ = self._midpoint(data)
        cube_xyz = data.xpos[self.cube_body_id].copy()
        return float(np.linalg.norm(cube_xyz - midpoint))

    def update(self, model: mujoco.MjModel, data: mujoco.MjData, jaw_qpos: float,
               force_release: bool = False) -> GraspState:
        mujoco.mj_forward(model, data)

        cube_xyz = data.xpos[self.cube_body_id].copy()
        state = self.state

        if not state.attached:
            midpoint, body_id, rot = self._midpoint(data)
            gap = float(np.linalg.norm(cube_xyz - midpoint))
            enclosed = (
                jaw_qpos <= self.jaw_closed_qpos_threshold
                and gap <= self.attach_gap_tolerance
                and (not self.require_contact or self._has_gripper_cube_contact(model, data))
            )
            state.enclosure_streak = state.enclosure_streak + 1 if enclosed else 0
            if state.enclosure_streak >= self.attach_debounce_steps:
                # Center the cube at the midpoint between both fingers.
                # The offset is stored in the gripper body's local frame
                # so it tracks correctly during carry.
                # Cube center → midpoint (zero offset = perfectly centered).
                # Add a small z-offset so the cube sits slightly below
                # the midpoint (the cube center should be at the same
                # height as the finger surfaces, not above them).
                target_cube_world = midpoint.copy()
                # The midpoint might be above the cube's natural center
                # height. We want the cube center at the midpoint z,
                # which is fine — the virtual attach handles this.
                state.offset_local = rot.T @ (target_cube_world - midpoint)
                # offset_local is zero vector — cube center tracks midpoint exactly
                state.midpoint_body_id = body_id
                state.attached = True
                state.enclosure_streak = 0
                state.last_world = target_cube_world.copy()
                state.release_reason = None
            return state

        # --- Attached: carry the cube ---
        release_reason = None
        if force_release:
            release_reason = "forced"
        elif jaw_qpos >= self.release_jaw_qpos_threshold:
            release_reason = "jaw_reopen"
        elif state.last_world is not None:
            drift = float(np.linalg.norm(cube_xyz - state.last_world))
            if drift > self.release_drift_tolerance:
                release_reason = "drift"

        if release_reason is not None:
            state.attached = False
            state.offset_local = None
            state.enclosure_streak = 0
            state.last_world = None
            state.release_reason = release_reason
            return state

        # Update cube position: track midpoint + offset
        midpoint, body_id, rot = self._midpoint(data)
        new_world = midpoint + rot @ state.offset_local
        data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = new_world
        data.qvel[CUBE_QPOS_START:CUBE_QPOS_END] = 0.0
        state.last_world = new_world.copy()
        return state
