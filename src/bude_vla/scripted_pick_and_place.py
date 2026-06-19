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
from bude_vla.grasp import GraspController, JAW_CLOSED_QPOS_THRESHOLD, IK_SEED_JAW_QPOS
from bude_vla.envs.so101_mjx import (
    ARM_QPOS_START, ARM_QPOS_END, GRIPPER_QPOS_START, CUBE_QPOS_START,
)


APPROACH = 0
GRASP = 1
LIFT = 2
MOVE = 3
RELEASE = 4

WRIST_FLEX_LOCK = np.pi / 2
WRIST_ROLL_LOCK = 0.0

GROUND_Z = 0.0295
BALL_RADIUS = 0.0125
HOVER_ABOVE_BALL = 0.10
LIFT_ABOVE_TARGET = 0.18
DROP_EE_Z = 0.07
JAW_OPEN = 1.5
JAW_CLOSED = -0.175

GRASP_DESCEND_STEPS = 60
# Slow enough that the moving finger sweeps past the ball faster than the ball
# can roll out of the cup under lateral pressure (empirically, ~60 was too fast).
GRASP_GRIP_STEPS = 120
GRASP_HOLD_STEPS = 50
GRASP_TIMEOUT_STEPS = 350
# Past this jaw position the actuator is mechanically seated on whatever the
# moving finger is pressing against. Empirical: with the ball in cup, jaw
# ctrl=-0.175 drives qpos down to ~1.0 (not 0) within ~30 substeps, then
# plateaus because the cup stops the finger.
JAW_DEEMED_CLOSED = 1.0
LIFT_RAMP_STEPS = 80
MOVE_RAMP_STEPS = 120


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
            arm_target[3] = WRIST_FLEX_LOCK
            arm_target[4] = WRIST_ROLL_LOCK
            cur = data.qpos[ARM_QPOS_START:ARM_QPOS_END].astype(np.float64).copy()
            frac = min(1.0, self.phase_step / 50.0)
            tgt = self._interp_qpos(cur, arm_target.astype(np.float64), frac)
            tgt[3] = WRIST_FLEX_LOCK
            tgt[4] = WRIST_ROLL_LOCK
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
            # Land the jaw tip 12mm above the ball top. Closer than 12mm pushes
            # the gripper body near the bowl rim (adds stray `cube` contacts);
            # further than 12mm leaves the finger too high for the 5mm attach
            # gap tolerance, so closure never even gates ATTACHED.
            jaw_target = np.array([
                live_ball_xy[0],
                live_ball_xy[1],
                GROUND_Z + BALL_RADIUS + 0.012,
            ])
            seed_qpos = data.qpos.copy()
            seed_qpos[GRIPPER_QPOS_START] = IK_SEED_JAW_QPOS
            arm_target = _ik_core(
                self.model, self.jaw_site_id, jaw_target, seed_qpos,
                step=0.5, damping=0.05, pos_tol=0.003, max_iters=150,
            ).astype(np.float64)
            arm_target[3] = WRIST_FLEX_LOCK
            arm_target[4] = WRIST_ROLL_LOCK

            cur = data.qpos[ARM_QPOS_START:ARM_QPOS_END].astype(np.float64).copy()

            if self.phase_step <= GRASP_DESCEND_STEPS:
                # DESCEND: arm ramps down to grasp pose, jaw stays FULLY OPEN.
                # Closing the jaw here was causing the moving finger to sweep
                # the ball out of the cup before the arm arrived at the pose.
                frac = min(1.0, self.phase_step / GRASP_DESCEND_STEPS)
                next_arm = self._interp_qpos(cur, arm_target, frac)
                next_arm[3] = WRIST_FLEX_LOCK
                next_arm[4] = WRIST_ROLL_LOCK
                data.qpos[ARM_QPOS_START:ARM_QPOS_END] = next_arm
                data.qvel[ARM_QPOS_START:ARM_QPOS_END] = 0
                ctrl[GRIPPER_QPOS_START] = JAW_OPEN
                if self.phase_step == GRASP_DESCEND_STEPS:
                    self._grasp_arm_q = arm_target.copy()
            elif self.phase_step <= GRASP_DESCEND_STEPS + GRASP_GRIP_STEPS:
                # GRIP: arm pinned at the descended pose, jaw drives closed.
                data.qpos[ARM_QPOS_START:ARM_QPOS_END] = self._grasp_arm_q
                data.qvel[ARM_QPOS_START:ARM_QPOS_END] = 0
                grip_step = self.phase_step - GRASP_DESCEND_STEPS
                frac = min(1.0, grip_step / float(GRASP_GRIP_STEPS))
                ctrl[GRIPPER_QPOS_START] = JAW_OPEN - frac * (JAW_OPEN - JAW_CLOSED)
            else:
                # HOLD: arm + jaw fully commanded, wait for actuator to seat +
                # GraspController's enclosure/contact gates to fire.
                data.qpos[ARM_QPOS_START:ARM_QPOS_END] = self._grasp_arm_q
                data.qvel[ARM_QPOS_START:ARM_QPOS_END] = 0
                ctrl[GRIPPER_QPOS_START] = JAW_CLOSED

            jaw_qpos = float(data.qpos[GRIPPER_QPOS_START])
            self.grasp.update(model, data, jaw_qpos=jaw_qpos)

            hold_complete = self.phase_step >= GRASP_DESCEND_STEPS + GRASP_GRIP_STEPS + GRASP_HOLD_STEPS
            jaw_closed_enough = jaw_qpos <= JAW_DEEMED_CLOSED
            if self.grasp.state.attached and jaw_closed_enough and hold_complete:
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
            # Aim straight up from the CURRENT end-effector position so the
            # arm travels vertically above the carried ball instead of
            # swinging sideways back toward the original cube_start_xy.
            current_ee = data.site_xpos[self.site_id].copy()
            target = np.array([
                current_ee[0],
                current_ee[1],
                GROUND_Z + LIFT_ABOVE_TARGET,
            ])
            arm_target = self._ik_target(data, target).astype(np.float64)
            arm_target[3] = WRIST_FLEX_LOCK
            arm_target[4] = WRIST_ROLL_LOCK
            cur = data.qpos[ARM_QPOS_START:ARM_QPOS_END].astype(np.float64).copy()
            if self.phase_step == 1:
                self._lift_arm_q = arm_target
            frac = min(1.0, self.phase_step / float(LIFT_RAMP_STEPS))
            data.qpos[ARM_QPOS_START:ARM_QPOS_END] = self._interp_qpos(cur, arm_target, frac)
            data.qvel[ARM_QPOS_START:ARM_QPOS_END] = 0
            ctrl[GRIPPER_QPOS_START] = JAW_CLOSED
            if self.phase_step > LIFT_RAMP_STEPS + 20:
                self.phase = MOVE
                self.phase_step = 0

        elif self.phase == MOVE:
            target = np.array([
                self.target_xy[0],
                self.target_xy[1],
                GROUND_Z + LIFT_ABOVE_TARGET,
            ])
            arm_target = self._ik_target(data, target).astype(np.float64)
            arm_target[3] = WRIST_FLEX_LOCK
            arm_target[4] = WRIST_ROLL_LOCK
            cur = data.qpos[ARM_QPOS_START:ARM_QPOS_END].astype(np.float64).copy()
            frac = min(1.0, self.phase_step / float(MOVE_RAMP_STEPS))
            data.qpos[ARM_QPOS_START:ARM_QPOS_END] = self._interp_qpos(cur, arm_target, frac)
            data.qvel[ARM_QPOS_START:ARM_QPOS_END] = 0
            ctrl[GRIPPER_QPOS_START] = JAW_CLOSED
            if self.phase_step > MOVE_RAMP_STEPS + 20:
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
