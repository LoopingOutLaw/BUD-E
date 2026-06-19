"""Scripted pick-and-place: approach -> descend -> close jaw -> carry -> move -> release.

Kinematic arm control (qpos override via IK) + dynamic jaw control + ball
teleport during carry. The ball starts inside a constrained pick bowl so
the gripper descent cannot push it out laterally. Once the jaw closes
around the ball, its world-frame position is computed in gripper-local
coordinates and maintained each step during LIFT/MOVE/RELEASE phases.
"""
from __future__ import annotations
import numpy as np
import mujoco
from bude_vla.ik import solve_ik_to_xyz_dls
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
GRASP_EE_Z_OFFSET = 0.000
LIFT_ABOVE_TARGET = 0.18
DROP_EE_Z = 0.07
JAW_OPEN = 1.5
JAW_CLOSED = -0.175


class ScriptedPickAndPlace:
    def __init__(self, model, data, cube_start_xy, target_xy=(0.30, 0.40)):
        self.model = model
        self.cube_start_xy = np.asarray(cube_start_xy, dtype=np.float64)
        self.target_xy = np.asarray(target_xy, dtype=np.float64)
        self.phase = APPROACH
        self.phase_step = 0
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self.cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._max_steps = 600
        self._total_steps = 0
        self._cube_attached_offset = None

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

    def _carry_ball(self, data):
        if self._cube_attached_offset is None:
            return
        gripper_xyz = data.xpos[self.gripper_body_id].copy()
        gripper_rot = data.xmat[self.gripper_body_id].reshape(3, 3).copy()
        new_cube_world = gripper_xyz + gripper_rot @ self._cube_attached_offset
        data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = new_cube_world
        data.qvel[CUBE_QPOS_START:CUBE_QPOS_START + 6] = 0

    def _attach_offset(self, data):
        gripper_xyz = data.xpos[self.gripper_body_id].copy()
        gripper_rot = data.xmat[self.gripper_body_id].reshape(3, 3).copy()
        cube_xyz = self._cube_xyz(data)
        self._cube_attached_offset = gripper_rot.T @ (cube_xyz - gripper_xyz)

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
            if self.phase_step > 80 or self.phase_step > 0 and self.phase_step % 1 == 0 and self.phase_step >= 50:
                ball_drift = np.linalg.norm(ball[:2] - self.cube_start_xy)
                if ball_drift < 0.02:
                    self.phase = GRASP
                    self.phase_step = 0
                elif self.phase_step > 90:
                    self.phase = GRASP
                    self.phase_step = 0

        elif self.phase == GRASP:
            target = np.array([
                self.cube_start_xy[0],
                self.cube_start_xy[1],
                GROUND_Z + BALL_RADIUS + 0.020,
            ])
            arm_target = self._ik_target(data, target).astype(np.float64)
            if self.phase_step == 1:
                self._grasp_arm_q = arm_target
            cur = data.qpos[ARM_QPOS_START:ARM_QPOS_END].astype(np.float64).copy()
            if self.phase_step < 60:
                frac = min(1.0, self.phase_step / 50.0)
                data.qpos[ARM_QPOS_START:ARM_QPOS_END] = self._interp_qpos(cur, arm_target, frac)
                data.qvel[ARM_QPOS_START:ARM_QPOS_END] = 0
                ctrl[GRIPPER_QPOS_START] = JAW_OPEN
            else:
                data.qpos[ARM_QPOS_START:ARM_QPOS_END] = self._grasp_arm_q
                data.qvel[ARM_QPOS_START:ARM_QPOS_END] = 0
                jaw = JAW_OPEN - ((self.phase_step - 60) / 60.0) * (JAW_OPEN - JAW_CLOSED)
                jaw = max(jaw, JAW_CLOSED)
                ctrl[GRIPPER_QPOS_START] = jaw
                if self.phase_step == 100:
                    self._attach_offset(data)
            if self.phase_step > 130:
                self.phase = LIFT
                self.phase_step = 0

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
            self._carry_ball(data)
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
            self._carry_ball(data)
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
            ctrl[GRIPPER_QPOS_START] = JAW_CLOSED
            self._carry_ball(data)
            if self.phase_step > 70:
                self._cube_attached_offset = None
                ctrl[GRIPPER_QPOS_START] = JAW_OPEN
            if self.phase_step > 110:
                done = True

        if self._total_steps >= self._max_steps:
            done = True

        return ctrl, data.qpos[ARM_QPOS_START:ARM_QPOS_END].copy(), done, {"phase": self.phase, "phase_step": self.phase_step}
