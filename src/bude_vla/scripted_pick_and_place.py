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

from collections.abc import Callable

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
BACKOFF = 6

# Gripper params matching ggand0/pick-101 exactly
GRIPPER_OPEN = 0.3
GRIPPER_CLOSED = -0.8
GRIPPER_TIGHTEN = 0.4

# Wrist angles for top-down approach (matching pick-101)
WRIST_FLEX_LOCK = np.pi / 2
WRIST_ROLL_LOCK = np.pi / 2

# 3cm cube
CUBE_HALF_WIDTH = 0.015
FINGER_WIDTH_OFFSET = -0.015
GRASP_Z_OFFSET = 0.01
HEIGHT_OFFSET = 0.03

# Step counts
APPROACH_STEPS = 100
DESCENT_STEPS = 100
CLOSE_STEPS = 300
TIGHTEN_STEPS = 100
LIFT_STEPS = 100
HOLD_STEPS = 50
MOVE_STEPS = 300
RELEASE_STEPS = 100
BACKOFF_STEPS = 70
NUDGE_DESCENT_STEPS = 65
GRASP_CONTACT_GRACE_STEPS = 8

PHASE_NAMES = {
    0: "APPROACH", 1: "DESCENT", 2: "CLOSE", 3: "LIFT",
    4: "MOVE", 5: "RELEASE", 6: "BACKOFF",
}


def decaying_recovery_offset(offset_xy: np.ndarray, phase_step: int, total_steps: int) -> np.ndarray:
    """Linearly remove recovery jitter so final grasp targets the real cube."""
    if total_steps <= 0:
        return np.zeros_like(offset_xy, dtype=np.float64)
    frac = min(max(float(phase_step) / float(total_steps), 0.0), 1.0)
    return np.asarray(offset_xy, dtype=np.float64) * (1.0 - frac)


def decaying_recovery_scalar(offset: float, phase_step: int, total_steps: int) -> float:
    """Scalar version of decaying recovery jitter for depth/height demos."""
    if total_steps <= 0:
        return 0.0
    frac = min(max(float(phase_step) / float(total_steps), 0.0), 1.0)
    return float(offset) * (1.0 - frac)


def should_retry_close(contact_step: int | None, retries_used: int, max_retries: int) -> bool:
    return contact_step is None and retries_used < max(0, max_retries)


def has_recent_grasp_contact(
    phase_step: int,
    last_contact_step: int | None,
    grace_steps: int = GRASP_CONTACT_GRACE_STEPS,
) -> bool:
    return last_contact_step is not None and phase_step - last_contact_step <= grace_steps


class ScriptedPickAndPlace:
    """Physics-only pick-and-place using ggand0/pick-101's proven approach.

    No GraspController, no kinematic carry, no teleport. The finger pads
    + friction hold the cube naturally through contact dynamics.
    """

    def __init__(self, model, data, cube_start_xy, target_xy=(0.32, 0.16),
                 recovery_jitter_xy: float = 0.0,
                 recovery_jitter_z: float = 0.0,
                 recovery_jitter_prob: float = 0.0,
                 max_grasp_retries: int = 0,
                 nudge_recovery_prob: float = 0.0,
                 nudge_recovery_xy: float = 0.0,
                 nudge_recovery_z: float = 0.0,
                 retry_miss_xy: float = 0.0,
                 retry_miss_prob: float = 0.0,
                 cube_position_provider: Callable[[object], np.ndarray] | None = None,
                 rng: np.random.Generator | None = None):
        self.model = model
        self._cube_position_provider = cube_position_provider
        self.cube_start_xy = np.asarray(cube_start_xy, dtype=np.float64)
        self.target_xy = np.asarray(target_xy, dtype=np.float64)
        self.phase = APPROACH
        self.phase_step = 0
        self._total_steps = 0
        self._max_steps = 2200

        rng = rng if rng is not None else np.random.default_rng()
        use_recovery = recovery_jitter_prob > 0.0 and (
            recovery_jitter_xy > 0.0 or recovery_jitter_z > 0.0
        ) and rng.random() < recovery_jitter_prob
        if use_recovery:
            self._approach_recovery_xy = rng.uniform(
                -recovery_jitter_xy, recovery_jitter_xy, size=2)
            self._descent_recovery_xy = rng.uniform(
                -recovery_jitter_xy, recovery_jitter_xy, size=2)
            self._descent_recovery_z = float(rng.uniform(
                -recovery_jitter_z, recovery_jitter_z))
        else:
            self._approach_recovery_xy = np.zeros(2, dtype=np.float64)
            self._descent_recovery_xy = np.zeros(2, dtype=np.float64)
            self._descent_recovery_z = 0.0

        self.max_grasp_retries = max(0, int(max_grasp_retries))
        self._retries_used = 0
        self._nudge_recovery_enabled = (
            nudge_recovery_prob > 0.0
            and (nudge_recovery_xy > 0.0 or nudge_recovery_z > 0.0)
            and rng.random() < nudge_recovery_prob
        )
        self._nudge_recovery_done = False
        self._nudge_touched = False
        if self._nudge_recovery_enabled:
            self._nudge_recovery_xy = rng.uniform(
                -nudge_recovery_xy, nudge_recovery_xy, size=2)
            self._nudge_recovery_z = -abs(float(rng.uniform(
                0.0, nudge_recovery_z)))
        else:
            self._nudge_recovery_xy = np.zeros(2, dtype=np.float64)
            self._nudge_recovery_z = 0.0

        use_retry_miss = (
            self.max_grasp_retries > 0
            and retry_miss_xy > 0.0
            and rng.random() < retry_miss_prob
        )
        if use_retry_miss:
            self._close_retry_miss_xy = rng.uniform(-retry_miss_xy, retry_miss_xy, size=2)
        else:
            self._close_retry_miss_xy = np.zeros(2, dtype=np.float64)

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
        self._last_grasp_contact_step = None
        self._grasp_action = GRIPPER_CLOSED
        self._grasp_succeeded = False

        # Store cube position at start (for approach target)
        self._cube_start_xyz = np.array([cube_start_xy[0], cube_start_xy[1], CUBE_REST_Z])
        self._grasp_anchor_xyz = self._cube_start_xyz.copy()

    # ---- helpers ----

    def _cube_xyz(self, data):
        if self._cube_position_provider is not None:
            position = np.asarray(self._cube_position_provider(data), dtype=np.float64)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError(
                    "cube_position_provider must return one finite xyz position"
                )
            return position.copy()
        return data.xpos[self.cube_body_id].copy()

    def _capture_grasp_anchor(self, data) -> None:
        """Remember where the cube was grasped, including retry displacement."""
        self._grasp_anchor_xyz = self._cube_xyz(data)

    def _reacquire_cube_if_available(self, data) -> None:
        reacquire = getattr(self._cube_position_provider, "reacquire", None)
        if callable(reacquire):
            reacquire(data)


    def _cube_pad_contacts(self, data) -> set[int]:
        contacts = set()
        for i in range(data.ncon):
            g1, g2 = data.contact[i].geom1, data.contact[i].geom2
            if g1 == self.cube_geom_id or g2 == self.cube_geom_id:
                other = g2 if g1 == self.cube_geom_id else g1
                contacts.add(other)
        return contacts

    def is_touching_cube(self, model, data):
        """Any finger-pad contact with the cube, including failed nudges."""
        contacts = self._cube_pad_contacts(data)
        return self.static_pad_id in contacts or self.moving_pad_id in contacts

    def is_grasping(self, model, data):
        """Contact-based grasp detection — both finger pads touching cube."""
        contacts = self._cube_pad_contacts(data)
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
            if self._retries_used > 0 and self.phase_step == 35:
                self._reacquire_cube_if_available(data)

            cube_live = self._cube_xyz(data)
            above_pos = cube_live.copy()
            above_pos[2] += GRASP_Z_OFFSET + HEIGHT_OFFSET
            above_pos[1] += FINGER_WIDTH_OFFSET
            above_pos[:2] += decaying_recovery_offset(
                self._approach_recovery_xy, self.phase_step, APPROACH_STEPS)

            ctrl = self.ik.step_toward_target(
                above_pos, gripper_action=GRIPPER_OPEN,
                gain=0.8, locked_joints=locked_joints,
            )

            if self.phase_step >= APPROACH_STEPS:
                self.phase = DESCENT
                self.phase_step = 0

        elif self.phase == DESCENT:
            cube_live = self._cube_xyz(data)
            grasp_target = cube_live.copy()
            grasp_target[2] += GRASP_Z_OFFSET
            grasp_target[1] += FINGER_WIDTH_OFFSET
            if self._nudge_recovery_enabled and not self._nudge_recovery_done:
                grasp_target[:2] += self._nudge_recovery_xy
                grasp_target[2] += self._nudge_recovery_z
            else:
                grasp_target[:2] += decaying_recovery_offset(
                    self._descent_recovery_xy, self.phase_step, DESCENT_STEPS)
                grasp_target[2] += decaying_recovery_scalar(
                    self._descent_recovery_z, self.phase_step, DESCENT_STEPS)

            ctrl = self.ik.step_toward_target(
                grasp_target, gripper_action=GRIPPER_OPEN,
                gain=0.8, locked_joints=locked_joints,
            )

            if (
                self._nudge_recovery_enabled
                and not self._nudge_recovery_done
                and (self.is_touching_cube(model, data) or self.phase_step >= NUDGE_DESCENT_STEPS)
            ):
                self._nudge_touched = self._nudge_touched or self.is_touching_cube(model, data)
                self.phase = BACKOFF
                self.phase_step = 0
            elif self.phase_step >= DESCENT_STEPS:
                self.phase = CLOSE
                self.phase_step = 0

        elif self.phase == BACKOFF:
            backoff_pos = self._cube_xyz(data).copy()
            backoff_pos[2] += GRASP_Z_OFFSET + HEIGHT_OFFSET + 0.025
            backoff_pos[1] += FINGER_WIDTH_OFFSET

            ctrl = self.ik.step_toward_target(
                backoff_pos, gripper_action=GRIPPER_OPEN,
                gain=0.7, locked_joints=locked_joints,
            )

            if self.phase_step >= BACKOFF_STEPS:
                self._nudge_recovery_done = True
                self.phase = DESCENT
                self.phase_step = 0

        elif self.phase == CLOSE:
            # Contact-triggered closing (matching pick-101 exactly)
            grasp_target = self._cube_xyz(data).copy()
            grasp_target[2] += GRASP_Z_OFFSET
            grasp_target[1] += FINGER_WIDTH_OFFSET
            if self._retries_used == 0:
                grasp_target[:2] += self._close_retry_miss_xy

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

            # A brief contact flicker is not a grasp. Keep only contact that
            # persists through the tightening window (with a small physics grace).
            grasping_now = self.is_grasping(model, data)
            if grasping_now:
                self._last_grasp_contact_step = self.phase_step
                if self._contact_step is None:
                    self._contact_step = self.phase_step
                    self._contact_action = gripper
            elif not has_recent_grasp_contact(
                self.phase_step, self._last_grasp_contact_step
            ):
                self._contact_step = None
                self._contact_action = None

            contact_recent = has_recent_grasp_contact(
                self.phase_step, self._last_grasp_contact_step
            )

            # Check if done tightening.
            if self._contact_step is not None:
                target_action = max(self._contact_action - GRIPPER_TIGHTEN, -1.0)
                if gripper <= target_action + 0.01 and contact_recent:
                    self._grasp_action = gripper
                    self._grasp_succeeded = True
                    self._capture_grasp_anchor(data)
                    self.phase = LIFT
                    self.phase_step = 0

            if self.phase_step >= CLOSE_STEPS:
                if self._contact_step is not None and contact_recent:
                    self._grasp_action = gripper
                    self._grasp_succeeded = True
                    self._capture_grasp_anchor(data)
                    self.phase = LIFT
                    self.phase_step = 0
                elif should_retry_close(self._contact_step, self._retries_used, self.max_grasp_retries):
                    self._retries_used += 1
                    self._contact_step = None
                    self._contact_action = None
                    self._last_grasp_contact_step = None
                    self.phase = APPROACH
                    self.phase_step = 0
                else:
                    done = True

        elif self.phase == LIFT:
            grasp_target = self._grasp_anchor_xyz.copy()
            grasp_target[2] += GRASP_Z_OFFSET
            grasp_target[1] += FINGER_WIDTH_OFFSET
            lift_pos = self._grasp_anchor_xyz.copy()
            lift_pos[2] += HEIGHT_OFFSET + 0.05
            lift_pos[1] += FINGER_WIDTH_OFFSET

            t = min(self.phase_step / float(LIFT_STEPS), 1.0)
            target = grasp_target + (lift_pos - grasp_target) * t

            ctrl = self.ik.step_toward_target(
                target, gripper_action=self._grasp_action,
                gain=0.3, locked_joints=locked_joints,
            )

            if self.phase_step >= LIFT_STEPS + HOLD_STEPS:
                self.phase = MOVE
                self.phase_step = 0

        elif self.phase == MOVE:
            lift_pos = self._grasp_anchor_xyz.copy()
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
            "retries_used": self._retries_used,
            "nudge_recovery": self._nudge_recovery_enabled,
            "nudge_touched": self._nudge_touched,
        }
