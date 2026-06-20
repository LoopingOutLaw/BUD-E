"""Diagnostic: measure gripper geometry — pure MuJoCo (no JAX).

Uses MjSpec builder (same path as so101_mjx._build_composite_spec)
to build the model, avoiding XML loading issues with includes.

Run:
    MUJOCO_GL=egl PYTHONPATH=src python scripts/diagnose_gripper_geom.py
"""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mujoco
import numpy as np

# Build model via MjSpec (same as so101_mjx but without JAX dependency)
from pathlib import Path

_ARM_SPEC_PATH = (
    Path(__file__).resolve().parents[1] / "urdf" / "so101_official" / "so101_new_calib.xml"
)

CUBE_XY = np.array([0.30, 0.0])
CUBE_Z_CENTER = 0.010

HOME_Q = np.array([0.0, -0.5, 0.95, -0.55, 0.0])

ARM_QPOS_START = 0
ARM_QPOS_END = 5
GRIPPER_QPOS_START = 5
CUBE_QPOS_START = 6

WRIST_FLEX_LOCK = np.pi / 2
WRIST_ROLL_LOCK = 0.0

GROUP_DEFAULT = 1
GROUP_BALL = 2
GROUP_BOWL = 4
BALL_CONTYPE = GROUP_DEFAULT | GROUP_BALL
BALL_CONAFFINITY = GROUP_DEFAULT | GROUP_BALL
BOWL_CONTYPE = GROUP_BOWL
BOWL_CONAFFINITY = GROUP_BALL

import math as _m


def _add_ring_wall(parent_body, radius, wall_height, z_center, n_segments, thickness,
                   rgba, contype, conaffinity, name_prefix, overlap=1.18):
    circumference = 2 * _m.pi * radius
    seg_full_width = (circumference / n_segments) * overlap
    half_width = seg_full_width / 2.0
    for i in range(n_segments):
        ang = 2 * _m.pi * i / n_segments
        parent_body.add_geom(
            name=f"{name_prefix}_{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[thickness, half_width, wall_height / 2.0],
            pos=[radius * _m.cos(ang), radius * _m.sin(ang), z_center],
            euler=[0, 0, ang + _m.pi / 2],
            rgba=rgba, contype=contype, conaffinity=conaffinity,
            condim=3 if contype != 0 else 1,
        )


def build_model():
    cwd = os.getcwd()
    arm_dir = _ARM_SPEC_PATH.parent
    os.chdir(arm_dir)
    try:
        spec = mujoco.MjSpec.from_file(_ARM_SPEC_PATH.name)
        spec.option.timestep = 0.002
        spec.option.iterations = 100
        spec.option.ls_iterations = 50
        spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
        spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL

        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[1, 1, 0.05], rgba=[0.7, 0.7, 0.8, 1], condim=3,
            friction=[1.0, 0.005, 0.0001])
        spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, 0.25, 0.0], size=[0.35, 0.10, 0.02],
            rgba=[0.85, 0.78, 0.65, 1], condim=3)

        ball = spec.worldbody.add_body(name="cube", pos=[0.30, 0.0, 0.010])
        ball.add_joint(name="cube_free", type=mujoco.mjtJoint.mjJNT_FREE)
        ball.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.010, 0.010, 0.010], rgba=[0.85, 0.05, 0.05, 1],
            mass=0.30, condim=6, friction=[5.0, 0.5, 0.1],
            solref=[0.01, 1.0], solimp=[0.95, 0.99, 0.001, 0.5, 2],
            contype=BALL_CONTYPE, conaffinity=BALL_CONAFFINITY)

        tgt = spec.worldbody.add_body(name="target_zone", pos=[0.32, 0.16, 0.021])
        tgt.add_geom(name="target_zone_disc", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.06, 0.06, 0.002], rgba=[0.1, 0.3, 0.95, 1],
            contype=0, conaffinity=0)
        tgt.add_geom(name="target_zone_inner", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.025, 0.025, 0.003], rgba=[0.95, 0.95, 1.0, 1],
            pos=[0, 0, 0.0005], contype=0, conaffinity=0)

        bowl = spec.worldbody.add_body(name="bowl", pos=[0.32, 0.16, 0.016])
        bowl.add_geom(name="bowl_floor", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.032, 0.002], rgba=[0.30, 0.30, 0.36, 1],
            contype=BOWL_CONTYPE, conaffinity=BOWL_CONAFFINITY, condim=6,
            friction=[5.0, 0.5, 0.1])
        _add_ring_wall(bowl, radius=0.038, wall_height=0.040, z_center=0.020,
            n_segments=24, thickness=0.003, rgba=[0.25, 0.25, 0.30, 1],
            contype=BOWL_CONTYPE, conaffinity=BOWL_CONAFFINITY, name_prefix="bowl_rim")

        # Fixed finger on gripper body
        gripper_body = next(
            (b for b in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY) if b.name == "gripper"), None)
        if gripper_body is not None:
            gripper_body.add_geom(name="fixed_finger", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.008, 0.012, 0.010], pos=[-0.022, -0.006, -0.004],
                rgba=[0.2, 0.8, 0.2, 0.5], friction=[5.0, 0.5, 0.1], condim=6)
            gripper_body.add_site(name="fixed_finger_contact",
                pos=[-0.022, -0.006, -0.004], size=0.005, rgba=[0.3, 1, 0.3, 0.6])

        # jaw_contact site on moving_jaw body
        jaw_body = next(
            (b for b in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY) if b.name == "moving_jaw_so101_v1"), None)
        if jaw_body is not None:
            jaw_body.add_site(name="jaw_contact",
                pos=[-0.001, -0.025, 0.019], size=0.005, rgba=[1, 0.3, 0.3, 0.6])

        return spec.compile()
    finally:
        os.chdir(cwd)


def _ik_core(model, site_id, target_xyz, current_qpos, step=0.5, damping=0.05,
             pos_tol=0.001, max_iters=100):
    """Damped least-squares IK for the 5 arm joints."""
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    qpos = current_qpos.copy()
    data = mujoco.MjData(model)

    for _ in range(max_iters):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        ee_xyz = data.site_xpos[site_id]
        err = target_xyz - ee_xyz
        if np.linalg.norm(err) < pos_tol:
            break

        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = jacp[:, ARM_QPOS_START:ARM_QPOS_END]

        JJt = J @ J.T
        dq_arm = step * J.T @ np.linalg.solve(JJt + (damping**2) * np.eye(JJt.shape[0]), err)
        qpos[ARM_QPOS_START:ARM_QPOS_END] += dq_arm
        qpos[ARM_QPOS_START:ARM_QPOS_END] = np.clip(qpos[ARM_QPOS_START:ARM_QPOS_END], -np.pi, np.pi)

    return qpos[ARM_QPOS_START:ARM_QPOS_END].copy()


def main():
    model = build_model()

    site_gf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    site_jc = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "jaw_contact")
    site_ff = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fixed_finger_contact")
    body_gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    body_mjaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")

    gripper_geoms = []
    for i in range(model.ngeom):
        bid = model.geom_bodyid[i]
        if bid == body_gripper or bid == body_mjaw:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
            gripper_geoms.append((i, name, bid))

    print("Gripper/moving_jaw geoms:")
    for gid, name, bid in gripper_geoms:
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        gtype_num = model.geom_type[gid]
        gtype_name = {mujoco.mjtGeom.mjGEOM_BOX: "BOX", mujoco.mjtGeom.mjGEOM_MESH: "MESH"}
        tn = gtype_name.get(gtype_num, f"TYPE_{gtype_num}")
        sz = model.geom_size[gid]
        pos = model.geom_pos[gid]
        print(f"  geom {gid}: '{name}' on body '{bname}' type={tn} pos={pos} size={sz}")
    print()

    # ---- Part 1: Jaw angle vs finger separation ----
    print("=== Jaw angle vs finger separation (home pose, no descent) ===")
    for jaw_angle in np.arange(-0.175, 1.8, 0.2):
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        data.qpos[:5] = HOME_Q
        data.qpos[GRIPPER_QPOS_START] = jaw_angle
        data.qpos[CUBE_QPOS_START:CUBE_QPOS_START+3] = [CUBE_XY[0], CUBE_XY[1], CUBE_Z_CENTER]
        data.qpos[CUBE_QPOS_START+3:CUBE_QPOS_START+7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

        jc = data.site_xpos[site_jc].copy()
        ff = data.site_xpos[site_ff].copy() if site_ff >= 0 else np.array([0,0,-999])
        gf = data.site_xpos[site_gf].copy()

        if site_ff >= 0:
            dist_3d = float(np.linalg.norm(jc - ff))
            # Vertical (z) separation between fingertips and gripperframe
            jc_below_gf = gf[2] - jc[2]
            ff_below_gf = gf[2] - ff[2]
        else:
            dist_3d = -999
            jc_below_gf = gf[2] - jc[2]
            ff_below_gf = -999

        target_125x = 1.25 * 2 * 0.010  # 0.025 m
        marker = " ← ~1.25×" if abs(dist_3d - target_125x) < 0.005 else ""

        print(f"  jaw={jaw_angle:+7.3f}  gf_z={gf[2]:.4f}  jc_z={jc[2]:.4f}  ff_z={ff[2]:.4f}  "
              f"jc_below_gf={jc_below_gf:.4f}  ff_below_gf={ff_below_gf:.4f}  "
              f"grip_width={dist_3d:.4f}{marker}")

    # ---- Part 2: Descent height analysis ----
    print("\n=== Descent height: gripperframe z vs fingertip positions ===")
    jaw_angles_descent = [-0.175, 0.0, 0.3, 0.5]
    gf_z_targets = [0.12, 0.10, 0.08, 0.07, 0.065, 0.06, 0.055, 0.05, 0.048, 0.04, 0.038, 0.036, 0.034, 0.03]

    for jaw_angle in jaw_angles_descent:
        print(f"\n--- Jaw angle: {jaw_angle:.3f} rad ---")
        print(f"  {'gf_tgt':>8}  {'gf_act':>8}  {'jc_z':>8}  {'ff_z':>8}  {'lowest':>8}  {'safe?':>6}  {'cube_gap':>8}")

        for gf_z in gf_z_targets:
            data = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)
            data.qpos[:5] = HOME_Q
            data.qpos[GRIPPER_QPOS_START] = 1.5
            data.qpos[CUBE_QPOS_START:CUBE_QPOS_START+3] = [CUBE_XY[0], CUBE_XY[1], CUBE_Z_CENTER]
            data.qpos[CUBE_QPOS_START+3:CUBE_QPOS_START+7] = [1.0, 0.0, 0.0, 0.0]
            mujoco.mj_forward(model, data)

            target_xyz = np.array([CUBE_XY[0], CUBE_XY[1], gf_z])
            seed = data.qpos.copy()
            seed[GRIPPER_QPOS_START] = jaw_angle
            arm_q = _ik_core(model, site_gf, target_xyz, seed, step=0.5, damping=0.05,
                             pos_tol=0.001, max_iters=100)
            data.qpos[ARM_QPOS_START:ARM_QPOS_END] = arm_q
            data.qpos[3] = WRIST_FLEX_LOCK
            data.qpos[4] = WRIST_ROLL_LOCK
            data.qpos[GRIPPER_QPOS_START] = jaw_angle
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)

            gf_actual = float(data.site_xpos[site_gf][2])
            jc_z = float(data.site_xpos[site_jc][2]) if site_jc >= 0 else -999
            ff_z = float(data.site_xpos[site_ff][2]) if site_ff >= 0 else -999

            lowest_z = 999.0
            for gid, name, bid in gripper_geoms:
                geom_z = float(data.geom_xpos[gid][2])
                gtype = model.geom_type[gid]
                if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                    half_z = float(model.geom_size[gid][2])
                    lowest_z = min(lowest_z, geom_z - half_z)
                elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
                    rbound = float(model.geom_rbound[gid])
                    body_z = float(data.xpos[bid][2])
                    lowest_z = min(lowest_z, body_z - rbound)
                else:
                    lowest_z = min(lowest_z, geom_z)

            # Cube top is at z=0.020, cube center at z=0.010
            # gap = how far the lower fingertip (min(jc_z, ff_z)) is from cube top
            fingertip_z = min(jc_z, ff_z) if jc_z > -999 and ff_z > -999 else jc_z
            cube_gap = fingertip_z - 0.020  # positive = fingertip above cube top

            safe = "OK" if lowest_z > 0.0 else "PEN"
            print(f"  {gf_z:8.4f}  {gf_actual:8.4f}  {jc_z:8.4f}  {ff_z:8.4f}  {lowest_z:8.4f}  {safe:>6}  {cube_gap:8.4f}")


if __name__ == "__main__":
    main()
