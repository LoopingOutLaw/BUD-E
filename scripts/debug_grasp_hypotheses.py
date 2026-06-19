#!/usr/bin/env python
"""Single-shot diagnostic to resolve four hypotheses about the grasp bug.

H1: IK target z=GROUND_Z=0.0295 puts the gripper base INTO the table.
H2: jaw-open axis points UP not sideways — can't enclose the ball horizontally.
H3: 'attached=True' fires on non-ball contacts (table, bowl rim).
H4: orientation constraint over-constrains 5-DOF arm (already mostly confirmed
    via ori_weight sweep — will light it up cleanly here too).

Run with: unset PYTHONPATH && /home/aditya/venv-bude/bin/python scripts/debug_grasp_hypotheses.py
Output:    /tmp/bude_debug/probe_<X>.txt  +  stdout summary.
"""
import os, sys
sys.path.insert(0, '/home/aditya/bude_vla/src')
os.environ['MUJOCO_GL'] = 'egl'

import json
import time
from pathlib import Path

import numpy as np
import mujoco

from bude_vla.envs.so101_mjx import load_arm_model
from bude_vla.ik import _ik_core

OUT = Path('/tmp/bude_debug')
OUT.mkdir(parents=True, exist_ok=True)


def ik_target_for_ball(m, d, ball_xy, z_offset):
    """Solve IK with jaw_contact at (ball_xy, GROUND_Z + z_offset). Returns qpos."""
    target = np.array([ball_xy[0], ball_xy[1], 0.0295 + z_offset], dtype=np.float64)
    seed = d.qpos.copy().astype(np.float64)
    seed[5] = 0.30  # IK_SEED_JAW_QPOS
    return _ik_core(m, 3, target, seed,
                    step=0.5, damping=0.05, pos_tol=0.003, max_iters=150)


def snapshot_pose(m, d, label, f):
    """After IK solve + mj_forward, dump key world positions to f."""
    f.write(f'\n=== {label} ===\n')
    f.write(f'jaw_contact (site 3) = {np.array2string(d.site_xpos[3], precision=5)}\n')
    f.write(f'gripper (body 6)     = {np.array2string(d.xpos[6], precision=5)}\n')
    f.write(f'moving_jaw (body 7)  = {np.array2string(d.xpos[7], precision=5)}\n')
    f.write(f'ball (body 9)        = {np.array2string(d.xpos[9], precision=5)}\n')
    f.write(f'table top z          = 0.0 +0.02 = 0.020 (geom pos + half-size)\n')
    f.write(f'  gripper.z - table_top = {d.xpos[6][2] - 0.020:.4f} m (NEG = into table)\n')
    f.write(f'  jaw.z     - ball.z    = {d.xpos[7][2] - d.xpos[9][2]:+.4f} m\n')
    f.write(f'  jaw.xy   - ball.xy    = {np.linalg.norm(d.xpos[7][:2] - d.xpos[9][:2]):.4f} m (lateral miss)\n')
    # jaw-open axis in world frame (local X of moving_jaw body)
    R = d.xmat[7].reshape(3, 3)
    f.write(f'jaw open axis +X_world = {R[:, 0]}  |z|={abs(R[2,0]):.3f}\n')
    f.write(f'jaw length  +Y_world  = {R[:, 1]}  |z|={abs(R[2,1]):.3f}\n')


def main():
    m = load_arm_model()
    d = mujoco.MjData(m)

    # Probe 1: IK target at ball center (current behavior, line 119 of
    # scripted_pick_and_place.py GRASP phase)
    with open(OUT / 'probe_H1_target_too_low.txt', 'w') as f:
        f.write('IEEE: H1 — IK target at ball center z=0.0295 is TOO LOW relative to gripper base\n')
        ball_xy = np.array([0.305, -0.009])
        # set ball to the cube position of episode 0 from verify run
        q = ik_target_for_ball(m, d, ball_xy, z_offset=0.0)  # current target
        d.qpos[0:5] = q
        mujoco.mj_forward(m, d)
        snapshot_pose(m, d, 'IK target at ball CENTER (z=0.0295)', f)
        # also try the ABOVE-the-ball pose, like HOVER_ABOVE_BALL = 0.10
        q_above = ik_target_for_ball(m, d, ball_xy, z_offset=+0.10)
        d.qpos[0:5] = q_above
        mujoco.mj_forward(m, d)
        snapshot_pose(m, d, 'IK target z=GROUND+0.10 (HOVER_ABOVE_BALL)', f)
        # Run a full step to see if the arm penetrates anything
        f.write('\n=== After mj_step(0.01s) from above pose ===\n')
        for _ in range(20):
            mujoco.mj_step(m, d)
        f.write(f'gripper.z   = {d.xpos[6][2]:.4f}  (table_top = 0.02)\n')
        f.write(f'moving_jaw.z = {d.xpos[7][2]:.4f}\n')
        f.write(f'ball.z      = {d.xpos[9][2]:.4f}\n')
        # any contacts?
        contacts = []
        for i in range(d.ncon):
            c = d.contact[i]
            gb1 = m.geom_bodyid[c.geom1]
            gb2 = m.geom_bodyid[c.geom2]
            contacts.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                             mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)))
        f.write(f'contacts (geom_pair): {contacts}\n')

    # Probe 2: jaw-open axis
    with open(OUT / 'probe_H2_jaw_axis.txt', 'w') as f:
        f.write('IEEE: H2 — jaw-open axis must point DOWN for proper ball enclosure\n')
        # Solve IK with the *correct* vertical-descending approach
        # (target above ball, jaw tip pointing down)
        f.write('\n=== sphere xpos after IK target above ball ===\n')
        # at default q_a, where is the jaw-open axis?
        R = d.xmat[7].reshape(3, 3)
        f.write(f'default jaw open axis world = {R[:,0]}\n')
        f.write(f'default jaw length  axis world = {R[:,1]}\n')
        f.write('  (length axis along world -X means the jaw "points" along -X)\n')
        f.write('  (open axis world+~Z means opening sideways along Y or vertical — depends on URDF pose)\n')

    # Probe 3: contact reports at "attach" verdict
    with open(OUT / 'probe_H3_false_attach.txt', 'w') as f:
        from bude_vla.scripted_pick_and_place import (
            ScriptedPickAndPlace, HOVER_ABOVE_BALL, GRASP, APPROACH,
            IK_SEED_JAW_QPOS, JAW_OPEN, GROUND_Z,
        )
        from bude_vla.grasp import GraspController
        f.write('IEEE: H3 — does the "attached=True" predicate fire on non-ball contacts?\n')
        for seed in range(3):
            np.random.seed(seed)
            cx, cy = np.random.uniform(0.28, 0.32), np.random.uniform(-0.02, 0.02)
            plan = ScriptedPickAndPlace(m, d, (cx, cy))
            # Pre-load qpos a little so the ball is settled
            d.qpos[6] = cx
            d.qpos[7] = cy
            d.qpos[8] = 0.030
            mujoco.mj_forward(m, d)
            # Take 230 steps (full APPROACH + GRASP phase), dump contacts at every 10th
            f.write(f'\n--- seed={seed} cx={cx:.3f} cy={cy:.3f} ---\n')
            for step in range(230):
                ctrl, _, done, info = plan.step(m, d)
                # apply ctrl to actuators
                d.ctrl[:] = ctrl
                for _ in range(5):
                    mujoco.mj_step(m, d)
                if step % 10 == 0 or info.get('attached', False):
                    contacts = []
                    for i in range(d.ncon):
                        c = d.contact[i]
                        gb1 = m.geom_bodyid[c.geom1]
                        gb2 = m.geom_bodyid[c.geom2]
                        bname1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, gb1)
                        bname2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, gb2)
                        if ('jaw' in (bname1 or '') or 'jaw' in (bname2 or '')
                            or 'gripper' in (bname1 or '') or 'gripper' in (bname2 or '')):
                            contacts.append((bname1, bname2))
                    jaw_q = d.qpos[5]
                    gap = np.linalg.norm(d.site_xpos[3] - d.xpos[9]) - 0.0125
                    if info.get('attached', False) or gap < 0.012:
                        f.write(f'step {step:3d} phase {info["phase"]} ATTACH={info.get("attached")} jaw={jaw_q:.3f} gap={gap*1000:.1f}mm contacts={contacts}\n')

    print('Wrote probes to', OUT)


if __name__ == '__main__':
    main()
