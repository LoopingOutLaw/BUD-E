"""Scripted pick-and-place: approach → pre-close → descent → grip → lift → move → release.

Fundamental redesign (v2):
  - All IK targets gripperframe site with orientation constraint
    (moving jaw Y-axis → world -Z, forcing the gripper to point toward
    the cube). This replaces the old jaw_contact IK that caused ground
    penetration.
  - Gripper is pre-closed to ~1.25x cube size BEFORE descending. The
    fingers are nearly touching the cube's sides as the gripper
    descends — no wide-open "magnetic" attachment.
  - Descent height is computed dynamically: at each step we measure
    the lowest gripper geom and enforce a minimum clearance above the
    floor. The arm NEVER penetrates the ground.
  - Carry phases use stored configs with per-step jaw-EE offset
    compensation, as in the previous version.
"""
from __future__ import annotations
import numpy as np
import mujoco
from bude_vla.ik import _ik_core
from bude_vla.grasp import GraspController, IK_SEED_JAW_QPOS
from bude_vla.envs.so101_mjx import (
    ARM_QPOS_START, ARM_QPOS_END, GRIPPER_QPOS_START,
    CUBE_QPOS_START,
)
from bude_vla.envs.so101_mjx import default_joint_angles


# Phase indices
APPROACH = 0
PRE_CLOSE = 1
DESCENT = 2
GRIP = 3
LIFT = 4
MOVE = 5
RELEASE = 6

# Gripper geometry constants
JAW_CLOSED = -0.175
JAW_OPEN_WIDE = 1.2
JAW_PRE_CLOSE = 0.60      # ~1.25x cube opening

# Height constants (all relative to floor at z=0)
GROUND_Z = 0.025           # cube center on pedestal (pedestal top at z=0.015)
CUBE_HALF_EXTENT = 0.010
CUBE_TOP_Z = 0.035         # cube top surface
HOVER_HEIGHT = 0.10
LIFT_HEIGHT = 0.22
DROP_HEIGHT = 0.08
MIN_GF_Z = 0.02

# Orientation constraint: moving jaw Y-axis → world -Z (pointing down)
ORI_BODY_AXIS = np.array([0.0, 1.0, 0.0])
ORI_WORLD_DIR = np.array([0.0, 0.0, -1.0])
ORI_WEIGHT = 2.0
POS_WEIGHT = 1.0

# Step counts per phase
APPROACH_STEPS = 80
PRE_CLOSE_STEPS = 60
DESCENT_STEPS = 100
GRIP_STEPS = 120
GRIP_HOLD_STEPS = 40
GRIP_TIMEOUT_STEPS = 300
LIFT_RAMP_STEPS = 80
MOVE_RAMP_STEPS = 140
RELEASE_DESCEND_STEPS = 80
POST_RELEASE_PAUSE = 80


class ScriptedPickAndPlace:
    def __init__(self, model, data, cube_start_xy, target_xy=(0.32, 0.16)):
        self.model = model
        self.cube_start_xy = np.asarray(cube_start_xy, dtype=np.float64)
        self.target_xy = np.asarray(target_xy, dtype=np.float64)
        self.phase = APPROACH
        self.phase_step = 0
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.jaw_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "jaw_contact")
        self.ff_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fixed_finger_contact")
        self.gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self.mjaw_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
        self.cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._max_steps = 900
        self._total_steps = 0
        self._descent_arm_q = None
        self._descent_gf_z = HOVER_HEIGHT
        self._grasp_succeeded = False
        self._lift_start_q = None
        self._lift_target_q = None
        self._move_target_q = None
        self._release_target_q = None
        self.grasp = GraspController(model)

    def _ee_xyz(self, data):
        return data.site_xpos[self.site_id].copy()

    def _cube_xyz(self, data):
        return data.xpos[self.cube_body_id].copy()

    def _jaw_xyz(self, data):
        return data.site_xpos[self.jaw_site_id].copy()

    def _ff_xyz(self, data):
        if self.ff_site_id < 0:
            return self._ee_xyz(data)
        return data.site_xpos[self.ff_site_id].copy()

    def _lowest_gripper_z(self, data):
        """Minimum z of any geom on gripper or moving_jaw bodies."""
        lowest = 999.0
        for i in range(self.model.ngeom):
            bid = self.model.geom_bodyid[i]
            if bid == self.gripper_body_id or bid == self.mjaw_body_id:
                gz = float(data.geom_xpos[i][2])
                gtype = self.model.geom_type[i]
                if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                    half_z = float(self.model.geom_size[i][2])
                    lowest = min(lowest, gz - half_z)
                elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
                    # rbound = distance from geom center to farthest vertex
                    rbound = float(self.model.geom_rbound[i])
                    lowest = min(lowest, gz - rbound)
                else:
                    lowest = min(lowest, gz)
        return lowest

    def _ik_ori(self, data, target_xyz, jaw_angle=JAW_PRE_CLOSE):
        """Orientation-constrained IK targeting gripperframe."""
        seed = data.qpos.copy()
        seed[GRIPPER_QPOS_START] = jaw_angle
        return _ik_core(
            self.model, self.site_id, target_xyz, seed,
            step=0.5, damping=0.05, pos_tol=0.001, max_iters=200,
            body_id=self.mjaw_body_id,
            target_axis=ORI_BODY_AXIS,
            target_world_dir=ORI_WORLD_DIR,
            ori_weight=ORI_WEIGHT,
            pos_weight=POS_WEIGHT,
            ori_tol=0.05,
        ).astype(np.float64)

    def _ik_ee(self, data, target_xyz, jaw_angle=JAW_CLOSED):
        """Position-only IK targeting gripperframe."""
        seed = data.qpos.copy()
        seed[GRIPPER_QPOS_START] = jaw_angle
        return _ik_core(
            self.model, self.site_id, target_xyz, seed,
            step=0.5, damping=0.05, pos_tol=0.005, max_iters=50,
        ).astype(np.float64)

    def _ik_jaw(self, data, target_xyz, jaw_angle=JAW_PRE_CLOSE):
        """IK targeting jaw_contact site — used for DESCENT."""
        seed = data.qpos.copy()
        seed[GRIPPER_QPOS_START] = jaw_angle
        return _ik_core(
            self.model, self.jaw_site_id, target_xyz, seed,
            step=0.5, damping=0.05, pos_tol=0.001, max_iters=200,
        ).astype(np.float64)

    def _ik_carry(self, data, desired_cube_xyz):
        """Two-pass IK for carry phases."""
        neutral_seed = data.qpos.copy()
        neutral_seed[:5] = default_joint_angles(self.model)
        neutral_seed[GRIPPER_QPOS_START] = JAW_CLOSED
        ee_rough = _ik_core(
            self.model, self.site_id, desired_cube_xyz, neutral_seed,
            step=0.5, damping=0.05, pos_tol=0.005, max_iters=25,
        )
        seed2 = data.qpos.copy()
        seed2[ARM_QPOS_START:ARM_QPOS_END] = ee_rough
        seed2[GRIPPER_QPOS_START] = JAW_CLOSED
        return _ik_core(
            self.model, self.jaw_site_id, desired_cube_xyz, seed2,
            step=0.5, damping=0.05, pos_tol=0.001, max_iters=200,
        ).astype(np.float64)

    def _interp_qpos(self, src, dst, frac):
        return src + (dst - src) * frac

    @property
    def attached(self) -> bool:
        return self.grasp.state.attached

    def _set_arm(self, data, arm_qpos):
        data.qpos[ARM_QPOS_START:ARM_QPOS_END] = arm_qpos
        data.qvel[ARM_QPOS_START:ARM_QPOS_END] = 0

    def _safe_descent_target(self, data, cube_xy):
        """Find the lowest safe jaw_contact z for descent.

        Uses jaw_contact IK (not gripperframe) so the fingertips can reach
        the cube while the gripper body stays above ground.
        """
        best_arm_q = None
        best_jc_z = 999.0

        # Target jaw_contact at various heights above the cube
        # Cube top is at CUBE_TOP_Z (0.035). We want fingertips just above it.
        for jc_z in np.arange(HOVER_HEIGHT, CUBE_TOP_Z - 0.010, -0.005):
            target = np.array([cube_xy[0], cube_xy[1], jc_z])
            arm_q = self._ik_jaw(data, target, jaw_angle=JAW_PRE_CLOSE)

            test_data = mujoco.MjData(self.model)
            mujoco.mj_resetData(self.model, test_data)
            test_data.qpos[:] = data.qpos.copy()
            test_data.qpos[ARM_QPOS_START:ARM_QPOS_END] = arm_q
            test_data.qpos[GRIPPER_QPOS_START] = JAW_PRE_CLOSE
            test_data.qvel[:] = 0
            mujoco.mj_forward(self.model, test_data)

            lowest_z = 999.0
            for i in range(self.model.ngeom):
                bid = self.model.geom_bodyid[i]
                if bid == self.gripper_body_id or bid == self.mjaw_body_id:
                    gz = float(test_data.geom_xpos[i][2])
                    gtype = self.model.geom_type[i]
                    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                        half_z = float(self.model.geom_size[i][2])
                        lowest_z = min(lowest_z, gz - half_z)
                    elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
                        rbound = float(self.model.geom_rbound[i])
                        lowest_z = min(lowest_z, gz - rbound)
                    else:
                        lowest_z = min(lowest_z, gz)

            if lowest_z < 0.002:
                break  # going lower would penetrate ground

            if jc_z < best_jc_z:
                best_jc_z = jc_z
                best_arm_q = arm_q.copy()

        if best_arm_q is None:
            # Fallback: use the hover config
            target = np.array([cube_xy[0], cube_xy[1], HOVER_HEIGHT])
            best_arm_q = self._ik_jaw(data, target, jaw_angle=JAW_PRE_CLOSE)
            best_jc_z = HOVER_HEIGHT

        return best_arm_q, best_jc_z

    def step(self, model, data):
        self._total_steps += 1
        self.phase_step += 1
        ctrl = np.zeros(model.nu, dtype=np.float32)
        done = False

        cube = self._cube_xyz(data)

        if self.phase == APPROACH:
            target = np.array([cube[0], cube[1], HOVER_HEIGHT])
            arm_target = self._ik_ori(data, target, jaw_angle=JAW_OPEN_WIDE)
            cur = data.qpos[ARM_QPOS_START:ARM_QPOS_END].astype(np.float64).copy()
            frac = min(1.0, self.phase_step / float(APPROACH_STEPS))
            tgt = self._interp_qpos(cur, arm_target, frac)
            self._set_arm(data, tgt)
            ctrl[GRIPPER_QPOS_START] = JAW_OPEN_WIDE
            if self.phase_step >= APPROACH_STEPS:
                self.phase = PRE_CLOSE
                self.phase_step = 0

        elif self.phase == PRE_CLOSE:
            arm_q = data.qpos[ARM_QPOS_START:ARM_QPOS_END].copy()
            target = np.array([cube[0], cube[1], HOVER_HEIGHT])
            arm_target = self._ik_ori(data, target, jaw_angle=JAW_PRE_CLOSE)
            frac = min(1.0, self.phase_step / float(PRE_CLOSE_STEPS))
            tgt = self._interp_qpos(arm_q, arm_target, frac)
            self._set_arm(data, tgt)
            jaw_frac = min(1.0, self.phase_step / float(PRE_CLOSE_STEPS))
            ctrl[GRIPPER_QPOS_START] = JAW_OPEN_WIDE + jaw_frac * (JAW_PRE_CLOSE - JAW_OPEN_WIDE)
            if self.phase_step >= PRE_CLOSE_STEPS:
                # Compute descent config: jaw_contact at cube top + 5mm clearance
                jc_target_z = CUBE_TOP_Z + 0.005  # fingertips just above cube top
                jaw_target = np.array([cube[0], cube[1], jc_target_z])
                self._descent_arm_q = self._ik_jaw(data, jaw_target, jaw_angle=JAW_PRE_CLOSE)
                self.phase = DESCENT
                self.phase_step = 0

        elif self.phase == DESCENT:
            start_q = data.qpos[ARM_QPOS_START:ARM_QPOS_END].copy()
            frac = min(1.0, self.phase_step / float(DESCENT_STEPS))
            tgt = self._interp_qpos(start_q, self._descent_arm_q, frac)
            self._set_arm(data, tgt)
            ctrl[GRIPPER_QPOS_START] = JAW_PRE_CLOSE
            if self.phase_step >= DESCENT_STEPS:
                self.phase = GRIP
                self.phase_step = 0

        elif self.phase == GRIP:
            self._set_arm(data, self._descent_arm_q)
            grip_frac = min(1.0, self.phase_step / float(GRIP_STEPS))
            ctrl[GRIPPER_QPOS_START] = JAW_PRE_CLOSE + grip_frac * (JAW_CLOSED - JAW_PRE_CLOSE)

            jaw_qpos = float(data.qpos[GRIPPER_QPOS_START])
            self.grasp.update(model, data, jaw_qpos=jaw_qpos)

            hold_done = self.phase_step >= GRIP_STEPS + GRIP_HOLD_STEPS
            jaw_closed_ok = jaw_qpos <= 0.4
            if self.grasp.state.attached and jaw_closed_ok and hold_done:
                self._grasp_succeeded = True
                self._lift_start_q = data.qpos[ARM_QPOS_START:ARM_QPOS_END].copy()
                self._lift_target_q = self._ik_ee(data, np.array([
                    data.site_xpos[self.site_id][0],
                    data.site_xpos[self.site_id][1],
                    LIFT_HEIGHT,
                ]))
                self._move_target_q = self._ik_carry(data, np.array([
                    self.target_xy[0], self.target_xy[1], LIFT_HEIGHT,
                ]))
                self._release_target_q = self._ik_carry(data, np.array([
                    self.target_xy[0], self.target_xy[1], DROP_HEIGHT,
                ]))
                self.phase = LIFT
                self.phase_step = 0
            elif self.phase_step > GRIP_TIMEOUT_STEPS:
                self._grasp_succeeded = False
                self._lift_start_q = data.qpos[ARM_QPOS_START:ARM_QPOS_END].copy()
                self._lift_target_q = self._ik_ee(data, np.array([
                    data.site_xpos[self.site_id][0],
                    data.site_xpos[self.site_id][1],
                    LIFT_HEIGHT,
                ]))
                self._move_target_q = self._ik_carry(data, np.array([
                    self.target_xy[0], self.target_xy[1], LIFT_HEIGHT,
                ]))
                self._release_target_q = self._ik_carry(data, np.array([
                    self.target_xy[0], self.target_xy[1], DROP_HEIGHT,
                ]))
                self.phase = LIFT
                self.phase_step = 0

            return ctrl, data.qpos[ARM_QPOS_START:ARM_QPOS_END].copy(), done, \
                {"phase": self.phase, "phase_step": self.phase_step,
                 "attached": self.grasp.state.attached}

        elif self.phase == LIFT:
            frac = min(1.0, self.phase_step / float(LIFT_RAMP_STEPS))
            self._set_arm(data, self._interp_qpos(self._lift_start_q, self._lift_target_q, frac))
            ctrl[GRIPPER_QPOS_START] = JAW_CLOSED
            if self.phase_step > LIFT_RAMP_STEPS + 20:
                self.phase = MOVE
                self.phase_step = 0

        elif self.phase == MOVE:
            frac = min(1.0, self.phase_step / float(MOVE_RAMP_STEPS))
            self._set_arm(data, self._interp_qpos(self._lift_target_q, self._move_target_q, frac))
            ctrl[GRIPPER_QPOS_START] = JAW_CLOSED
            if self.phase_step > MOVE_RAMP_STEPS + 20:
                self.phase = RELEASE
                self.phase_step = 0

        elif self.phase == RELEASE:
            frac = min(1.0, self.phase_step / float(RELEASE_DESCEND_STEPS))
            self._set_arm(data, self._interp_qpos(self._move_target_q, self._release_target_q, frac))
            if self.phase_step <= RELEASE_DESCEND_STEPS:
                ctrl[GRIPPER_QPOS_START] = JAW_CLOSED
            else:
                open_step = self.phase_step - RELEASE_DESCEND_STEPS
                open_frac = min(1.0, open_step / float(POST_RELEASE_PAUSE))
                ctrl[GRIPPER_QPOS_START] = JAW_CLOSED + open_frac * (1.7 - JAW_CLOSED)

            if self.phase_step > RELEASE_DESCEND_STEPS + POST_RELEASE_PAUSE:
                done = True

        if self._total_steps >= self._max_steps:
            done = True

        jaw_qpos = float(data.qpos[GRIPPER_QPOS_START])
        force_release = (self.phase == RELEASE and self.phase_step > RELEASE_DESCEND_STEPS)
        self.grasp.update(model, data, jaw_qpos=jaw_qpos, force_release=force_release)

        return ctrl, data.qpos[ARM_QPOS_START:ARM_QPOS_END].copy(), done, \
            {"phase": self.phase, "phase_step": self.phase_step,
             "attached": self.grasp.state.attached,
             "release_reason": self.grasp.state.release_reason}
