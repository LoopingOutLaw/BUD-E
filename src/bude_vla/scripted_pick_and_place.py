"""Scripted pick-and-place policy: approach -> descend -> grip -> lift -> move -> release.

Uses kinematic arm control (direct qpos override from IK) and inverse-dynamics
finger action. Cube is carried in scripted state: once GRIP phase closes the
gripper and the EE is on the cube, the cube's qpos is set every step to match
the gripper's pose (relative offset captured at grip moment). On RELEASE the
cube goes free-fall again.
"""
from __future__ import annotations
import numpy as np
import mujoco
from bude_vla.ik import solve_ik_to_xyz_dls

APPROACH = 0
DESCEND = 1
GRIP = 2
LIFT = 3
MOVE = 4
RELEASE = 5
TABLE_Z = 0.42
CUBE_HALF = 0.025


class ScriptedPickAndPlace:
    def __init__(self, model, data, cube_start_xy, target_xy=(0.85, 0.0)):
        self.model = model
        self.cube_start_xy = np.asarray(cube_start_xy, dtype=np.float64)
        self.target_xy = np.asarray(target_xy, dtype=np.float64)
        self.phase = APPROACH
        self.phase_step = 0
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_center")
        self.gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self.cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_joint_adr = 0
        self._max_steps = 350
        self._total_steps = 0
        self._cube_attached_offset = None
        self._target_release_xy = np.asarray(target_xy, dtype=np.float64)

    def _ee_xyz(self, data):
        return data.site_xpos[self.site_id].copy()

    def _cube_xyz(self, data):
        return data.xpos[self.cube_body_id].copy()

    def _ik_target(self, data, target_xyz):
        return solve_ik_to_xyz_dls(
            self.model, data, target_xyz, data.qpos.copy(),
            step=0.5, damping=0.05, pos_tol=0.005, max_iters=25,
        )

    def _ctrl_from_target(self, data, target_qpos):
        ctrl = np.zeros(7, dtype=np.float32)
        for i in range(6):
            err = target_qpos[i] - data.qpos[7 + i]
            ctrl[i] = np.clip(err * 15.0, -1.0, 1.0)
        return ctrl

    def _attach_cube_to_gripper(self, data):
        """Capture cube center in gripper body's local frame so we can carry it."""
        gripper_xyz = data.xpos[self.gripper_body_id].copy()
        gripper_rot = data.xmat[self.gripper_body_id].reshape(3, 3).copy()
        cube_xyz = self._cube_xyz(data)
        local_xyz = gripper_rot.T @ (cube_xyz - gripper_xyz)
        self._cube_attached_offset = local_xyz

    def _carry_cube_with(self, data):
        """Override cube world position based on gripper's current pose + captured offset."""
        if self._cube_attached_offset is None:
            return
        gripper_xyz = data.xpos[self.gripper_body_id].copy()
        gripper_rot = data.xmat[self.gripper_body_id].reshape(3, 3).copy()
        new_cube_xyz = gripper_xyz + gripper_rot @ self._cube_attached_offset
        data.qpos[0:3] = new_cube_xyz

    def step(self, model, data):
        self._total_steps += 1
        self.phase_step += 1
        ctrl = np.zeros(7, dtype=np.float32)
        arm_target = data.qpos[7:13].copy()
        done = False
        cube = self._cube_xyz(data)
        ee = self._ee_xyz(data)

        if self.phase == APPROACH:
            goal = np.array([
                self.cube_start_xy[0],
                self.cube_start_xy[1],
                TABLE_Z + CUBE_HALF + 0.15,
            ])
            arm_target = self._ik_target(data, goal)
            ctrl = self._ctrl_from_target(data, arm_target)
            dist = np.linalg.norm(ee - goal)
            if dist < 0.04 or self.phase_step > 60:
                self.phase = DESCEND
                self.phase_step = 0

        elif self.phase == DESCEND:
            goal = np.array([
                self.cube_start_xy[0],
                self.cube_start_xy[1],
                TABLE_Z + CUBE_HALF + 0.06,
            ])
            arm_target = self._ik_target(data, goal)
            ctrl = self._ctrl_from_target(data, arm_target)
            dist = np.linalg.norm(ee - goal)
            if dist < 0.04 or self.phase_step > 50:
                self.phase = GRIP
                self.phase_step = 0

        elif self.phase == GRIP:
            goal = np.array([cube[0], cube[1], cube[2] + 0.03])
            arm_target = self._ik_target(data, goal)
            ctrl = self._ctrl_from_target(data, arm_target)
            ctrl[6] = 1.0
            if self.phase_step >= 5 and self._cube_attached_offset is None:
                self._attach_cube_to_gripper(data)
            if self.phase_step > 20:
                self.phase = LIFT
                self.phase_step = 0

        elif self.phase == LIFT:
            goal = np.array([cube[0], cube[1], TABLE_Z + 0.25])
            arm_target = self._ik_target(data, goal)
            ctrl = self._ctrl_from_target(data, arm_target)
            ctrl[6] = 1.0
            dist = np.linalg.norm(ee - goal)
            if dist < 0.05 or self.phase_step > 50:
                self.phase = MOVE
                self.phase_step = 0

        elif self.phase == MOVE:
            goal = np.array([self.target_xy[0], self.target_xy[1], TABLE_Z + 0.25])
            arm_target = self._ik_target(data, goal)
            ctrl = self._ctrl_from_target(data, arm_target)
            ctrl[6] = 1.0
            dist = np.linalg.norm(ee - goal)
            if dist < 0.05 or self.phase_step > 60:
                self.phase = RELEASE
                self.phase_step = 0

        elif self.phase == RELEASE:
            goal = np.array([
                self.target_xy[0], self.target_xy[1],
                TABLE_Z + CUBE_HALF + 0.04,
            ])
            arm_target = self._ik_target(data, goal)
            ctrl = self._ctrl_from_target(data, arm_target)
            ctrl[6] = -1.0
            if self.phase_step >= 5:
                self._cube_attached_offset = None
            if self.phase_step > 20:
                done = True

        if self._total_steps >= self._max_steps:
            done = True

        return ctrl, arm_target, done, {"phase": self.phase, "phase_step": self.phase_step}
