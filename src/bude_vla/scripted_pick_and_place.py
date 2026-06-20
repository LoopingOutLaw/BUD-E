"""Scripted pick-and-place for SO-101 — direct port from ggand0/pick-101.

Uses their proven working approach exactly:
  - IKController targeting gripperframe (fingertip TCP)
  - wrist_flex=π/2 + wrist_roll=π/2 locked for top-down approach
  - Gripper starts partially open (0.3), closes to -0.8
  - Contact-triggered closing (both finger pads touching cube)
  - Physics-only grasping — NO kinematic carry/teleport
  - 3cm cube on floor, mass=0.03, wood friction

Clean 5-step sequence:
  APPROACH(0) → DESCENT(1) → CLOSE(2) → LIFT(3) → MOVE(4) → RELEASE(5)
"""
from __future__ import annotations

import numpy as np
import mujoco

from bude_vla.ik import IKController
from bude_vla.envs.so101_mjx import (
    CUBE_QPOS_START, CUBE_REST_Z,
)

APPROACH = 0
DESCENT = 1
CLOSE = 2
LIFT = 3
MOVE = 4
RELEASE = 5

# Gripper params matching ggand0/pick-101 exactly
GRIPPER_OPEN = 0.3
GRIPPER_CLOSED = -0.8
GRIPPER_TIGHTEN = 0.4

# Wrist angles for top-down approach (matching pick-101)
WRIST_FLEX_LOCK = np.pi / 2
WRIST_ROLL_LOCK = np.pi / 2

# 3cm cube (matching pick-101)
CUBE_HALF_WIDTH = 0.015
FINGER_WIDTH_OFFSET = -0.015   # Y offset to center grip (pick-101 uses -0.015 for 3cm cube)
GRASP_Z_OFFSET = 0.005        # grip slightly above cube center
HEIGHT_OFFSET = 0.03          # 30mm above cube for approach

# Step counts (matching pick-101)
APPROACH_STEPS = 300
DESCENT_STEPS = 200
CLOSE_STEPS = 300
TIGHTEN_STEPS = 100
LIFT_STEPS = 300
HOLD_STEPS = 200
MOVE_STEPS = 300
RELEASE_STEPS = 200

PHASE_NAMES = {
    0: "APPROACH", 1: "DESCENT", 2: "CLOSE", 3: "LIFT",
    4: "MOVE", 5: "RELEASE",
}


class ScriptedPickAndPlace:
    """Physics-only pick-and-place using ggand0/pick-101's proven approach.

    No GraspController, no kinematic carry, no teleport. The finger pads
    + friction hold the cube naturally through contact dynamics.
    """

    def __init__(self, model, data, cube_start_xy, target_xy=(0.32, 0.16)):
        self.model = model
        self.cube_start_xy = np.asarray(cube_start_xy, dtype=np.float64)
        self.target_xy = np.asarray(target_xy, dtype=np.float64)
        self.phase = APPROACH
        self.phase_step = 0
        self._total_steps = 0
        self._max_steps = 2000

        # IK controller (matching pick-101: targets gripperframe)
        self.ik = IKController(model, data, end_effector_site="gripperframe")

        # Body/geom IDs for contact detection (matching pick-101)
        self.cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.static_pad_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad")
        self.moving_pad_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad")

        # Contact tracking
        self._contact_step = None
        self._contact_action = None
        self._grasp_action = GRIPPER_CLOSED
        self._grasp_succeeded = False

        # Store cube position at start (for approach target)
        self._cube_start_xyz = np.array([cube_start_xy[0], cube_start_xy[1], CUBE_REST_Z])

    # ---- helpers ----

    def _cube_xyz(self, data):
        return data.xpos[self.cube_body_id].copy()

    def is_grasping(self, model, data):
        """Contact-based grasp detection — both finger pads touching cube."""
        contacts = set()
        for i in range(data.ncon):
            g1, g2 = data.contact[i].geom1, data.contact[i].geom2
            if g1 == self.cube_geom_id or g2 == self.cube_geom_id:
                other = g2 if g1 == self.cube_geom_id else g1
                contacts.add(other)
        has_static = self.static_pad_id in contacts
        has_moving = self.moving_pad_id in contacts
        return has_static and has_moving

    @property
    def attached(self) -> bool:
        return self._grasp_succeeded

    # ---- phase logic ----

    def step(self, model, data):
        """One policy step: compute ctrl and return it. Recorder calls mj_step."""
        self._total_steps += 1
        self.phase_step += 1
        done = False

        locked_joints = [3, 4]  # lock wrist_flex and wrist_roll
        cube = self._cube_xyz(data)

        if self.phase == APPROACH:
            above_pos = self._cube_start_xyz.copy()
            above_pos[2] += GRASP_Z_OFFSET + HEIGHT_OFFSET
            above_pos[1] += FINGER_WIDTH_OFFSET

            ctrl = self.ik.step_toward_target(
                above_pos, gripper_action=GRIPPER_OPEN,
                gain=0.5, locked_joints=locked_joints,
            )

            if self.phase_step >= APPROACH_STEPS:
                self.phase = DESCENT
                self.phase_step = 0

        elif self.phase == DESCENT:
            grasp_target = self._cube_start_xyz.copy()
            grasp_target[2] += GRASP_Z_OFFSET
            grasp_target[1] += FINGER_WIDTH_OFFSET

            ctrl = self.ik.step_toward_target(
                grasp_target, gripper_action=GRIPPER_OPEN,
                gain=0.5, locked_joints=locked_joints,
            )

            if self.phase_step >= DESCENT_STEPS:
                self.phase = CLOSE
                self.phase_step = 0

        elif self.phase == CLOSE:
            # Contact-triggered closing (matching pick-101 exactly)
            grasp_target = self._cube_start_xyz.copy()
            grasp_target[2] += GRASP_Z_OFFSET
            grasp_target[1] += FINGER_WIDTH_OFFSET

            if self._contact_step is None:
                t = min(self.phase_step / 250, 1.0)
                gripper = GRIPPER_OPEN - 2.0 * t
            else:
                steps_since = self.phase_step - self._contact_step
                t_slow = min(steps_since / float(TIGHTEN_STEPS), 1.0)
                target_action = max(self._contact_action - GRIPPER_TIGHTEN, -1.0)
                gripper = self._contact_action + (target_action - self._contact_action) * t_slow

            ctrl = self.ik.step_toward_target(
                grasp_target, gripper_action=gripper,
                gain=0.5, locked_joints=locked_joints,
            )
            # Force wrist angles (matching pick-101)
            ctrl[3] = WRIST_FLEX_LOCK
            ctrl[4] = WRIST_ROLL_LOCK

            # Detect contact
            if self.is_grasping(model, data) and self._contact_step is None:
                self._contact_step = self.phase_step
                self._contact_action = gripper

            # Check if done tightening
            if self._contact_step is not None:
                target_action = max(self._contact_action - GRIPPER_TIGHTEN, -1.0)
                if gripper <= target_action + 0.01:
                    self._grasp_action = gripper
                    self._grasp_succeeded = True
                    self.phase = LIFT
                    self.phase_step = 0

            if self.phase_step >= CLOSE_STEPS:
                if self._contact_step is not None:
                    self._grasp_action = gripper
                    self._grasp_succeeded = True
                self.phase = LIFT
                self.phase_step = 0

        elif self.phase == LIFT:
            grasp_target = self._cube_start_xyz.copy()
            grasp_target[2] += GRASP_Z_OFFSET
            grasp_target[1] += FINGER_WIDTH_OFFSET
            lift_pos = self._cube_start_xyz.copy()
            lift_pos[2] += HEIGHT_OFFSET + 0.05
            lift_pos[1] += FINGER_WIDTH_OFFSET

            t = min(self.phase_step / 200, 1.0)
            target = grasp_target + (lift_pos - grasp_target) * t

            ctrl = self.ik.step_toward_target(
                target, gripper_action=self._grasp_action,
                gain=0.3, locked_joints=locked_joints,
            )

            if self.phase_step >= LIFT_STEPS + HOLD_STEPS:
                self.phase = MOVE
                self.phase_step = 0

        elif self.phase == MOVE:
            lift_pos = self._cube_start_xyz.copy()
            lift_pos[2] += HEIGHT_OFFSET + 0.05
            lift_pos[1] += FINGER_WIDTH_OFFSET
            move_pos = np.array([
                self.target_xy[0],
                self.target_xy[1] + FINGER_WIDTH_OFFSET,
                lift_pos[2],
            ])

            t = min(self.phase_step / 200, 1.0)
            target = lift_pos + (move_pos - lift_pos) * t

            ctrl = self.ik.step_toward_target(
                target, gripper_action=self._grasp_action,
                gain=0.3, locked_joints=locked_joints,
            )

            if self.phase_step >= MOVE_STEPS:
                self.phase = RELEASE
                self.phase_step = 0

        elif self.phase == RELEASE:
            drop_pos = np.array([
                self.target_xy[0],
                self.target_xy[1] + FINGER_WIDTH_OFFSET,
                0.08,
            ])
            open_frac = min(self.phase_step / 100.0, 1.0)
            gripper = self._grasp_action + open_frac * (1.0 - self._grasp_action)

            ctrl = self.ik.step_toward_target(
                drop_pos, gripper_action=gripper,
                gain=0.3, locked_joints=locked_joints,
            )

            if self.phase_step >= RELEASE_STEPS:
                done = True

        if self._total_steps >= self._max_steps:
            done = True

        arm_q = data.qpos[:5].copy()
        return ctrl, arm_q, done, {
            "phase": self.phase,
            "phase_step": self.phase_step,
            "grasping": self.is_grasping(model, data) if self.phase >= CLOSE else False,
            "attached": self._grasp_succeeded,
        }
