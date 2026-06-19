"""Physically-gated grasp/carry logic for the SO-101 ball-pick task.

A single GraspController, shared by both recorder and eval, that only
attaches when ALL of the following are true:

  1. The ball's surface is within `attach_gap_tolerance` of the jaw
     contact site (the tip of the moving finger in the jaw body frame).
  2. The jaw's ACTUAL simulated joint angle is closed past
     `jaw_closed_qpos_threshold`.
  3. MuJoCo's own contact array shows a real contact between the moving
     jaw body and the ball body this step.
  4. All three of the above hold for `attach_debounce_steps` consecutive
     steps.

At the moment of attach, the ball is snapped flush -- the stored offset
corresponds to EXACTLY ball_radius from the jaw contact site, along
whatever direction it was approached from.

Release is **explicit**, not inferred: ScriptedPickAndPlace calls
update() with `force_release=True` once it has entered the RELEASE phase
and the jaw-open command has had time to act.  The jaw-angle and drift
checks are demoted to catastrophe-only safety nets (8cm drift, jaw
near-fully-open at 1.48) that should not fire in normal operation.
The `release_reason` field in GraspState exposes which path actually
triggered a release, so a debug trace can distinguish "script ran its
intended RELEASE phase" from "jaw swung open too early" or "ball was
physically knocked out of the jaw".
"""
from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from bude_vla.envs.so101_mjx import CUBE_QPOS_START, CUBE_QPOS_END

BALL_RADIUS = 0.0125
ATTACH_GAP_TOLERANCE = 0.005        # 5 mm — catches the near-contact window
ATTACH_DEBOUNCE_STEPS = 3           # attach faster before the jaw pushes the ball away
JAW_CLOSED_QPOS_THRESHOLD = 1.40    # attach detector: catches the enclosing window
                                    #     ~1.4 → ~0.5; calibrated jaw qpos realistically
                                    #     plateaus at ~0.475 against the 12.5mm ball
IK_SEED_JAW_QPOS = 0.30             # seed used by GRASP-phase IK so the arm is shaped
                                    #     for a CLOSED jaw while approaching (avoids the
                                    #     96deg arc plowing the ball sideways). This is a
                                    #     different role from JAW_CLOSED_QPOS_THRESHOLD —
                                    #     the IK solver only cares about kinematic shape,
                                    #     not enclosing detection.
RELEASE_JAW_QPOS_THRESHOLD = 1.48  # near-fully-open only; not the release path anymore
RELEASE_DRIFT_TOLERANCE = 0.08     # 8cm -- catches a real physical disaster, not noise


@dataclasses.dataclass
class GraspState:
    attached: bool = False
    offset_local: np.ndarray | None = None
    enclosure_streak: int = 0
    last_world: np.ndarray | None = None
    release_reason: str | None = None   # "forced" | "jaw_reopen" | "drift" | None

    def reset(self) -> None:
        self.attached = False
        self.offset_local = None
        self.enclosure_streak = 0
        self.last_world = None
        self.release_reason = None


class GraspController:
    """One instance per episode (or reuse across episodes and call `.reset()`).

    Resolves body/site ids once at construction, then `update()` is called
    once per simulation step. It will:
      - do nothing if not attached and the enclosure conditions aren't met,
      - attach (with debounce) once they are,
      - carry the ball rigidly while attached and conditions still hold,
      - release the ball once the jaw reopens or drift exceeds tolerance.
    """

    def __init__(self, model: mujoco.MjModel,
                 jaw_site_name: str = "jaw_contact",
                 jaw_body_name: str = "moving_jaw_so101_v1",
                 cube_body_name: str = "cube",
                 ball_radius: float = BALL_RADIUS,
                 attach_gap_tolerance: float = ATTACH_GAP_TOLERANCE,
                 attach_debounce_steps: int = ATTACH_DEBOUNCE_STEPS,
                 jaw_closed_qpos_threshold: float = JAW_CLOSED_QPOS_THRESHOLD,
                 release_jaw_qpos_threshold: float = RELEASE_JAW_QPOS_THRESHOLD,
                 release_drift_tolerance: float = RELEASE_DRIFT_TOLERANCE,
                 require_contact: bool = True):
        self.jaw_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, jaw_site_name)
        self.jaw_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, jaw_body_name)
        self.cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cube_body_name)
        for name, bid, kind in [(jaw_site_name, self.jaw_site_id, "site"),
                                (jaw_body_name, self.jaw_body_id, "body"),
                                (cube_body_name, self.cube_body_id, "body")]:
            if bid < 0:
                raise ValueError(
                    f"GraspController: {kind} '{name}' not found in model. "
                    f"Check load_arm_model() / _build_composite_spec()."
                )
        self.ball_radius = ball_radius
        self.attach_gap_tolerance = attach_gap_tolerance
        self.attach_debounce_steps = attach_debounce_steps
        self.jaw_closed_qpos_threshold = jaw_closed_qpos_threshold
        self.release_jaw_qpos_threshold = release_jaw_qpos_threshold
        self.release_drift_tolerance = release_drift_tolerance
        self.require_contact = require_contact
        self.state = GraspState()

    def reset(self) -> None:
        self.state.reset()

    def _has_jaw_ball_contact(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        body_ids = model.geom_bodyid
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = body_ids[c.geom1], body_ids[c.geom2]
            if (b1 == self.jaw_body_id and b2 == self.cube_body_id) or \
               (b1 == self.cube_body_id and b2 == self.jaw_body_id):
                return True
        return False

    def gap(self, data: mujoco.MjData) -> float:
        """Surface-to-surface distance between the ball and the jaw contact site."""
        jaw_xyz = data.site_xpos[self.jaw_site_id]
        ball_xyz = data.xpos[self.cube_body_id]
        return float(np.linalg.norm(ball_xyz - jaw_xyz)) - self.ball_radius

    def update(self, model: mujoco.MjModel, data: mujoco.MjData, jaw_qpos: float,
               force_release: bool = False) -> GraspState:
        """Advance grasp bookkeeping by one step and carry the ball if held.

        Must be called AFTER this step's qpos/ctrl writes. Internally calls
        mj_forward first so that data.xpos/xmat/contact reflect whatever
        qpos the caller just wrote.

        force_release: when True, releases the ball immediately regardless
        of jaw angle or drift.  ScriptedPickAndPlace passes this during
        the RELEASE phase (after the jaw-open command has been issued).
        The jaw/drift checks are catastrophic-only fallbacks (8cm drift,
        near-fully-open jaw) that should never fire during normal operation.
        """
        mujoco.mj_forward(model, data)

        jaw_xyz = data.site_xpos[self.jaw_site_id].copy()
        jaw_rot = data.xmat[self.jaw_body_id].reshape(3, 3).copy()
        ball_xyz = data.xpos[self.cube_body_id].copy()
        state = self.state

        if not state.attached:
            gap = float(np.linalg.norm(ball_xyz - jaw_xyz)) - self.ball_radius
            enclosed = (
                jaw_qpos <= self.jaw_closed_qpos_threshold
                and gap <= self.attach_gap_tolerance
                and (not self.require_contact or self._has_jaw_ball_contact(model, data))
            )
            state.enclosure_streak = state.enclosure_streak + 1 if enclosed else 0
            if state.enclosure_streak >= self.attach_debounce_steps:
                direction = ball_xyz - jaw_xyz
                norm = float(np.linalg.norm(direction))
                direction = direction / norm if norm > 1e-9 else np.array([0.0, 0.0, -1.0])
                flush_world = jaw_xyz + direction * self.ball_radius
                state.offset_local = jaw_rot.T @ (flush_world - jaw_xyz)
                state.attached = True
                state.enclosure_streak = 0
                state.last_world = flush_world.copy()
                state.release_reason = None
            return state

        release_reason = None
        if force_release:
            release_reason = "forced"
        elif jaw_qpos >= self.release_jaw_qpos_threshold:
            release_reason = "jaw_reopen"
        elif state.last_world is not None:
            drift = float(np.linalg.norm(ball_xyz - state.last_world))
            if drift > self.release_drift_tolerance:
                release_reason = "drift"

        if release_reason is not None:
            state.attached = False
            state.offset_local = None
            state.enclosure_streak = 0
            state.last_world = None
            state.release_reason = release_reason
            return state

        new_world = jaw_xyz + jaw_rot @ state.offset_local
        data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = new_world
        data.qvel[CUBE_QPOS_START:CUBE_QPOS_END] = 0.0
        state.last_world = new_world.copy()
        return state
