"""Scripted pick-and-place: approach -> close jaw -> carry -> move -> release.

Kinematic arm control (qpos override via IK) + dynamic jaw control. Grasp
attach/carry/release is delegated to `bude_vla.grasp.GraspController`, which
only attaches once the ball is geometrically enclosed AND in real contact
with the jaw AND that's held for several consecutive steps (see grasp.py
for the full rationale) -- this replaces the old behavior of capturing
whatever gripper-ball offset existed the instant the jaw started closing,
which is what produced a visible floating gap during both demo recording
and VLA rollout.

The GRASP phase also now aims at the ball's LIVE position every step
(instead of `cube_start_xy`, which is the ball's position from before it
settled in the pick bowl) -- if the ball drifted even ~1cm while settling,
the old code would aim, and then close the jaw, around empty space next
to it.
"""
from __future__ import annotations
import numpy as np
import mujoco
from bude_vla.ik import solve_ik_to_xyz_dls, _ik_core
from bude_vla.grasp import GraspController
from bude_vla.envs.so101_mjx import (
    ARM_QPOS_START, ARM_QPOS_END, GRIPPER_QPOS_START, CUBE_QPOS_START,
)


APPROACH = 0
GRASP = 1
LIFT = 2
MOVE = 3
RELEASE = 4

GROUND_Z = 0.0295
BALL_RADIUS = 0.0125
HOVER_ABOVE_BALL = 0.10
LIFT_ABOVE_TARGET = 0.18
DROP_EE_Z = 0.07
JAW_OPEN = 1.5
JAW_CLOSED = -0.175

GRASP_RAMP_STEPS = 60
GRASP_TIMEOUT_STEPS = 170


class ScriptedPickAndPlace:
    def __init__(self, model, data, cube_start_xy, target_xy=(0.30, 0.40)):
        self.model = model
        self.cube_start_xy = np.asarray(cube_start_xy, dtype=np.float64)
        self.target_xy = np.asarray(target_xy, dtype=np.float64)
        self.phase = APPROACH
        self.phase_step = 0
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.jaw_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "jaw_contact")
        self.gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self.cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._max_steps = 600
        self._total_steps = 0
        self._grasp_arm_q = None
        self._grasp_succeeded = False
        self.grasp = GraspController(model)

    def _ee_xyz(self, data):
        return data.site_xpos[self.site_id].copy()

    def _cube_xyz(self, data):
        return data.xpos[self.cube_body_id].copy()

    def _ik_target(self, data, target_xyz):
        return solve_ik_to_xyz_dls(
            self.model, data, target_xyz, data.qpos.copy(),
            step=0.5, damping=0.05, pos_tol=0.005, max_iters=25,
        )

    def _interp_qpos(self, src, dst, frac):
        return src + (dst - src) * frac

    @property
    def attached(self) -> bool:
        """True iff the ball is currently (physically-gated) held."""
        return self.grasp.state.attached

    def step(self, model, data):
        self._total_steps += 1
        self.phase_step += 1
        ctrl = np.zeros(model.nu, dtype=np.float32)
        done = False

        ball = self._cube_xyz(data)

        if self.phase == APPROACH:
            target = np.array([
                self.cube_start_xy[0],
                self.cube_start_xy[1],
                GROUND_Z + HOVER_ABOVE_BALL,
            ])
            arm_target = self._ik_target(data, target)
            cur = data.qpos[ARM_QPOS_START:ARM_QPOS_END].astype(np.float64).copy()
            frac = min(1.0, self.phase_step / 50.0)
            tgt = self._interp_qpos(cur, arm_target.astype(np.float64), frac)
            data.qpos[ARM_QPOS_START:ARM_QPOS_END] = tgt
            data.qvel[ARM_QPOS_START:ARM_QPOS_END] = 0
            ctrl[GRIPPER_QPOS_START] = JAW_OPEN
            if self.phase_step >= 50:
                ball_drift = np.linalg.norm(ball[:2] - self.cube_start_xy)
                if ball_drift < 0.02 or self.phase_step > 90:
                    self.phase = GRASP
                    self.phase_step = 0

        elif self.phase == GRASP:
            live_ball_xy = ball[:2]
            jaw_target = np.array([
                live_ball_xy[0],
                live_ball_xy[1],
                GROUND_Z + BALL_RADIUS,
            ])
            arm_target = _ik_core(
                self.model, self.jaw_site_id, jaw_target, data.qpos.copy(),
                step=0.5, damping=0.05, pos_tol=0.003, max_iters=150,
            ).astype(np.float64)
            cur = data.qpos[ARM_QPOS_START:ARM_QPOS_END].astype(np.float64).copy()
            if self.phase_step <= GRASP_RAMP_STEPS:
                frac = min(1.0, self.phase_step / 50.0)
                data.qpos[ARM_QPOS_START:ARM_QPOS_END] = self._interp_qpos(cur, arm_target, frac)
                data.qvel[ARM_QPOS_START:ARM_QPOS_END] = 0
                ctrl[GRIPPER_QPOS_START] = JAW_OPEN
                if self.phase_step == GRASP_RAMP_STEPS:
                    # Freeze the hold pose using THIS frame's IK solve, i.e.
                    # wherever the ball actually is right now.
                    self._grasp_arm_q = data.qpos[ARM_QPOS_START:ARM_QPOS_END].copy()
            else:
                data.qpos[ARM_QPOS_START:ARM_QPOS_END] = self._grasp_arm_q
                data.qvel[ARM_QPOS_START:ARM_QPOS_END] = 0
                close_step = self.phase_step - GRASP_RAMP_STEPS
                jaw = JAW_OPEN - (close_step / 60.0) * (JAW_OPEN - JAW_CLOSED)
                jaw = max(jaw, JAW_CLOSED)
                ctrl[GRIPPER_QPOS_START] = jaw

            jaw_qpos = float(data.qpos[GRIPPER_QPOS_START])
            self.grasp.update(model, data, jaw_qpos=jaw_qpos)

            if self.grasp.state.attached:
                self._grasp_succeeded = True
                self.phase = LIFT
                self.phase_step = 0
            elif self.phase_step > GRASP_TIMEOUT_STEPS:
                # Genuinely missed -- don't fake it. Move on; the episode
                # will correctly fail the final success check downstream.
                self._grasp_succeeded = False
                self.phase = LIFT
                self.phase_step = 0
            return ctrl, data.qpos[ARM_QPOS_START:ARM_QPOS_END].copy(), done, \
                {"phase": self.phase, "phase_step": self.phase_step, "attached": self.grasp.state.attached}

        elif self.phase == LIFT:
            target = np.array([
                self.cube_start_xy[0],
                self.cube_start_xy[1],
                GROUND_Z + LIFT_ABOVE_TARGET,
            ])
            arm_target = self._ik_target(data, target).astype(np.float64)
            cur = data.qpos[ARM_QPOS_START:ARM_QPOS_END].astype(np.float64).copy()
            if self.phase_step == 1:
                self._lift_arm_q = arm_target
            frac = min(1.0, self.phase_step / 50.0)
            data.qpos[ARM_QPOS_START:ARM_QPOS_END] = self._interp_qpos(cur, arm_target, frac)
            data.qvel[ARM_QPOS_START:ARM_QPOS_END] = 0
            ctrl[GRIPPER_QPOS_START] = JAW_CLOSED
            if self.phase_step > 60:
                self.phase = MOVE
                self.phase_step = 0

        elif self.phase == MOVE:
            target = np.array([
                self.target_xy[0],
                self.target_xy[1],
                GROUND_Z + LIFT_ABOVE_TARGET,
            ])
            arm_target = self._ik_target(data, target).astype(np.float64)
            cur = data.qpos[ARM_QPOS_START:ARM_QPOS_END].astype(np.float64).copy()
            frac = min(1.0, self.phase_step / 80.0)
            data.qpos[ARM_QPOS_START:ARM_QPOS_END] = self._interp_qpos(cur, arm_target, frac)
            data.qvel[ARM_QPOS_START:ARM_QPOS_END] = 0
            ctrl[GRIPPER_QPOS_START] = JAW_CLOSED
            if self.phase_step > 90:
                self.phase = RELEASE
                self.phase_step = 0

        elif self.phase == RELEASE:
            target = np.array([
                self.target_xy[0],
                self.target_xy[1],
                DROP_EE_Z,
            ])
            arm_target = self._ik_target(data, target).astype(np.float64)
            cur = data.qpos[ARM_QPOS_START:ARM_QPOS_END].astype(np.float64).copy()
            frac = min(1.0, self.phase_step / 60.0)
            data.qpos[ARM_QPOS_START:ARM_QPOS_END] = self._interp_qpos(cur, arm_target, frac)
            data.qvel[ARM_QPOS_START:ARM_QPOS_END] = 0
            if self.phase_step > 70:
                ctrl[GRIPPER_QPOS_START] = JAW_OPEN   # physically re-open; GraspController
            else:                                      # will detect the real release itself.
                ctrl[GRIPPER_QPOS_START] = JAW_CLOSED
            if self.phase_step > 110:
                done = True

        if self._total_steps >= self._max_steps:
            done = True

        jaw_qpos = float(data.qpos[GRIPPER_QPOS_START])
        self.grasp.update(model, data, jaw_qpos=jaw_qpos)

        return ctrl, data.qpos[ARM_QPOS_START:ARM_QPOS_END].copy(), done, \
            {"phase": self.phase, "phase_step": self.phase_step, "attached": self.grasp.state.attached}
