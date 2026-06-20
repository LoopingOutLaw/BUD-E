"""Calibrate a GRASP arm pose, then freeze the arm there and slowly close the
jaw, logging gap/qpos/contact every step. This tests whether a *good* IK
solution can physically enclose the ball -- independent of the scripted
policy's ramp/freeze timing.

Run with:
    unset PYTHONPATH
    cd /home/aditya/bude_vla
    MUJOCO_GL=egl PYTHONPATH=src python scripts/calibrate_grasp.py
"""
from __future__ import annotations
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from bude_vla.envs.so101_mjx import (
    load_arm_model, default_joint_angles,
    ARM_QPOS_START, ARM_QPOS_END, GRIPPER_QPOS_START, CUBE_QPOS_START,
)
from bude_vla.ik import _ik_core
from bude_vla.grasp import GraspController, BALL_RADIUS

model = load_arm_model()
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
data.qpos[ARM_QPOS_START:ARM_QPOS_END] = default_joint_angles(model)
data.qpos[GRIPPER_QPOS_START] = 1.5
GROUND_Z = 0.025  # cube half-extent on world floor -- must match grasp.py + scripted_pick_and_place.py
ball_xyz = np.array([0.30, 0.0, GROUND_Z])
data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = ball_xyz
data.qpos[CUBE_QPOS_START + 3:CUBE_QPOS_START + 7] = [1, 0, 0, 0]
mujoco.mj_forward(model, data)

jaw_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "jaw_contact")
jaw_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")

# --- calibrate: drive jaw_contact to the ball's north pole ---
target = ball_xyz + np.array([0, 0, BALL_RADIUS])
for it in range(8):
    arm_q = _ik_core(model, jaw_site_id, target, data.qpos.copy(),
                      method="dls", step=0.5, damping=0.05,
                      pos_tol=0.001, max_iters=200)
    data.qpos[ARM_QPOS_START:ARM_QPOS_END] = arm_q
    mujoco.mj_forward(model, data)
    err = target - data.site_xpos[jaw_site_id]
    print(f"it={it} err_mm={np.linalg.norm(err)*1000:.2f}")
    target = target + err

calibrated_q = data.qpos[ARM_QPOS_START:ARM_QPOS_END].copy()
print("\ncalibrated arm_q =", repr(calibrated_q))

# Log orientation
R = data.xmat[jaw_body_id].reshape(3, 3)
print("jaw local +X -> world:", R[:, 0])
print("jaw local +Y -> world:", R[:, 1])
print("jaw local +Z -> world:", R[:, 2])

# --- freeze arm, slowly close jaw, log everything ---
data.qpos[ARM_QPOS_START:ARM_QPOS_END] = calibrated_q
data.qpos[GRIPPER_QPOS_START] = 1.5
mujoco.mj_forward(model, data)

grasp = GraspController(model)
JAW_OPEN, JAW_CLOSED = 1.5, -0.175

print("\nclosing jaw at frozen calibrated pose:")
for step in range(120):
    frac = min(1.0, step / 60.0)
    jaw_ctrl = JAW_OPEN - frac * (JAW_OPEN - JAW_CLOSED)
    data.qpos[ARM_QPOS_START:ARM_QPOS_END] = calibrated_q
    data.qvel[ARM_QPOS_START:ARM_QPOS_END] = 0
    data.ctrl[:] = 0
    data.ctrl[GRIPPER_QPOS_START] = jaw_ctrl
    mujoco.mj_step(model, data)
    data.qpos[ARM_QPOS_START:ARM_QPOS_END] = calibrated_q
    jaw_qpos = float(data.qpos[GRIPPER_QPOS_START])
    state = grasp.update(model, data, jaw_qpos=jaw_qpos)
    if step % 5 == 0 or state.attached:
        gap = grasp.gap(data)
        contact = grasp._has_jaw_ball_contact(model, data)
        print(f"  step={step:3d} jaw_ctrl={jaw_ctrl:+.3f} jaw_qpos={jaw_qpos:+.3f} "
              f"gap_mm={gap*1000:+.2f} contact={contact} attached={state.attached}")
    if state.attached:
        print("  -> ATTACHED"); break
else:
    print("  -> never attached in 120 steps")
