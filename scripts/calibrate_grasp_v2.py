"""Verify the closed-jaw-seed fix for the side-on grasp bug.

The original calibrate_grasp.py reveals that IK seeded with jaw=1.5 (open)
positions the moving jaw INSIDE the ball at the north pole. As the jaw
closes through its 96 deg arc, it sweeps laterally and shoves the ball
away (gap_mm goes from -7 to +151).

Fix: seed the IK solver with the jaw already at the ATTACH threshold
(0.30), and target the ball's CENTER (equator) rather than its north
pole. The arm pose is computed for a nearly-closed jaw, so closing from
approach keeps the jaw centered on the ball rather than plowing through.

Run with:
    unset PYTHONPATH
    cd /home/aditya/bude_vla
    MUJOCO_GL=egl PYTHONPATH=src python scripts/calibrate_grasp_v2.py
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
from bude_vla.grasp import GraspController, BALL_RADIUS, JAW_CLOSED_QPOS_THRESHOLD, IK_SEED_JAW_QPOS

model = load_arm_model()
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

data.qpos[ARM_QPOS_START:ARM_QPOS_END] = default_joint_angles(model)
data.qpos[GRIPPER_QPOS_START] = 1.5
GROUND_Z = 0.0295
ball_xyz = np.array([0.30, 0.0, GROUND_Z])
data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = ball_xyz
data.qpos[CUBE_QPOS_START + 3:CUBE_QPOS_START + 7] = [1, 0, 0, 0]
mujoco.mj_forward(model, data)

jaw_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "jaw_contact")
jaw_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")

# --- FIX: seed with jaw at the closed-pose IK value, target ball center ---
target = ball_xyz.copy()  # ball center (equator), not north pole
seed = data.qpos.copy()
seed[GRIPPER_QPOS_START] = IK_SEED_JAW_QPOS  # 0.30 — closed shape for IK geometry

for it in range(8):
    arm_q = _ik_core(model, jaw_site_id, target, seed,
                     method="dls", step=0.5, damping=0.05,
                     pos_tol=0.001, max_iters=200)
    data.qpos[ARM_QPOS_START:ARM_QPOS_END] = arm_q
    # Keep jaw at the IK seed pose (closed) during the mj_forward that
    # measures the next err — otherwise measuring against the wide-open
    # jaw pull the target into the wrong place on subsequent iterations.
    data.qpos[GRIPPER_QPOS_START] = IK_SEED_JAW_QPOS
    mujoco.mj_forward(model, data)
    err = target - data.site_xpos[jaw_site_id]
    print(f"it={it} err_mm={np.linalg.norm(err)*1000:.2f}")
    target = target + err

calibrated_q = data.qpos[ARM_QPOS_START:ARM_QPOS_END].copy()
print("\ncalibrated arm_q (with closed-jaw seed):", repr(calibrated_q))

R = data.xmat[jaw_body_id].reshape(3, 3)
print("jaw local +X -> world:", R[:, 0])
print("jaw local +Y -> world:", R[:, 1])
print("jaw local +Z -> world:", R[:, 2])

# --- Now freeze arm, start with jaw OPEN, slowly close ---
data.qpos[ARM_QPOS_START:ARM_QPOS_END] = calibrated_q
data.qpos[GRIPPER_QPOS_START] = 1.5
mujoco.mj_forward(model, data)

grasp = GraspController(model)
JAW_OPEN, JAW_CLOSED = 1.5, -0.175

print("\nclosing jaw at frozen calibrated pose (FIXED seed):")
attached_step = None
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
        attached_step = step
        print("  -> ATTACHED")
        break
else:
    print("  -> NEVER ATTACHED")

print(f"\nattached_step = {attached_step}")
print(f"final gap_mm = {grasp.gap(data)*1000:+.2f}")
