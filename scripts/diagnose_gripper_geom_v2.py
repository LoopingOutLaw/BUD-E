"""Diagnostic V2: orientation-constrained IK + correct fixed finger position.

Uses the _ik_core orientation constraint to force gripper pointing down.
Also measures where the wrist_roll_follower collision mesh extends to
(find the real fixed jaw fingertip position).

Run:
    MUJOCO_GL=egl PYTHONPATH=src python scripts/diagnose_gripper_geom_v2.py
"""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mujoco
import numpy as np
from pathlib import Path
import math as _m

_ARM_SPEC_PATH = Path(__file__).resolve().parents[1] / "urdf" / "so101_official" / "so101_new_calib.xml"

CUBE_XY = np.array([0.30, 0.0])
CUBE_Z_CENTER = 0.010
CUBE_Z_TOP = 0.020

ARM_QPOS_START = 0
ARM_QPOS_END = 5
GRIPPER_QPOS_START = 5
CUBE_QPOS_START = 6

GROUP_DEFAULT = 1
GROUP_BALL = 2
GROUP_BOWL = 4
BALL_CONTYPE = GROUP_DEFAULT | GROUP_BALL
BALL_CONAFFINITY = GROUP_DEFAULT | GROUP_BALL
BOWL_CONTYPE = GROUP_BOWL
BOWL_CONAFFINITY = GROUP_BALL


def _add_ring_wall(parent_body, radius, wall_height, z_center, n_segments, thickness,
                   rgba, contype, conaffinity, name_prefix, overlap=1.18):
    circumference = 2 * _m.pi * radius
    seg_full_width = (circumference / n_segments) * overlap
    half_width = seg_full_width / 2.0
    for i in range(n_segments):
        ang = 2 * _m.pi * i / n_segments
        parent_body.add_geom(
            name=f"{name_prefix}_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[thickness, half_width, wall_height / 2.0],
            pos=[radius * _m.cos(ang), radius * _m.sin(ang), z_center],
            euler=[0, 0, ang + _m.pi / 2],
            rgba=rgba, contype=contype, conaffinity=conaffinity,
            condim=3 if contype != 0 else 1,
        )


def build_model(with_fixed_finger=True, fixed_finger_pos=None):
    """Build model, optionally with fixed finger at custom position."""
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

        gripper_body = next(
            (b for b in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY) if b.name == "gripper"), None)

        jaw_body = next(
            (b for b in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY) if b.name == "moving_jaw_so101_v1"), None)

        # Add jaw_contact site on moving_jaw
        if jaw_body is not None:
            jaw_body.add_site(name="jaw_contact",
                pos=[-0.001, -0.025, 0.019], size=0.005, rgba=[1, 0.3, 0.3, 0.6])

        # Add fixed finger geom and site at custom position
        if with_fixed_finger and gripper_body is not None and fixed_finger_pos is not None:
            gripper_body.add_geom(name="fixed_finger", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.008, 0.012, 0.010], pos=fixed_finger_pos,
                rgba=[0.2, 0.8, 0.2, 0.5], friction=[5.0, 0.5, 0.1], condim=6)
            gripper_body.add_site(name="fixed_finger_contact",
                pos=fixed_finger_pos, size=0.005, rgba=[0.3, 1, 0.3, 0.6])

        return spec.compile()
    finally:
        os.chdir(cwd)


def _ik_core(model, site_id, target_xyz, current_qpos, *,
             body_id=None, target_axis=None, target_world_dir=None,
             ori_weight=2.0, pos_weight=1.0,
             step=0.5, damping=0.05, pos_tol=0.001, max_iters=100):
    """DLS IK with optional orientation constraint."""
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    qpos = current_qpos.copy()
    data = mujoco.MjData(model)

    use_ori = (body_id is not None and target_axis is not None and target_world_dir is not None)
    if use_ori:
        target_axis_body = np.asarray(target_axis, dtype=np.float64)
        target_axis_body /= max(np.linalg.norm(target_axis_body), 1e-12)
        target_world_dir_unit = np.asarray(target_world_dir, dtype=np.float64)
        norm = np.linalg.norm(target_world_dir_unit)
        target_world_dir_unit = target_world_dir_unit / norm if norm > 1e-12 else np.array([1, 0, 0])

    for _ in range(max_iters):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        ee_xyz = data.site_xpos[site_id]
        err_pos = target_xyz - ee_xyz
        pos_err_norm = np.linalg.norm(err_pos)

        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J_pos = jacp[:, ARM_QPOS_START:ARM_QPOS_END]

        if use_ori:
            R = data.xmat[body_id].reshape(3, 3)
            current_dir = R @ target_axis_body
            err_ori = np.cross(target_world_dir_unit, current_dir)
            ori_err_norm = np.linalg.norm(err_ori)
            err = np.concatenate([err_pos * pos_weight, err_ori * ori_weight])
            J = np.vstack([J_pos * pos_weight, jacr[:, ARM_QPOS_START:ARM_QPOS_END] * ori_weight])
        else:
            ori_err_norm = 0.0
            err = err_pos
            J = J_pos

        if pos_err_norm < pos_tol and ori_err_norm < 0.05:
            break

        JJt = J @ J.T
        dq_arm = step * J.T @ np.linalg.solve(JJt + (damping**2) * np.eye(JJt.shape[0]), err)
        qpos[ARM_QPOS_START:ARM_QPOS_END] += dq_arm
        qpos[ARM_QPOS_START:ARM_QPOS_END] = np.clip(qpos[ARM_QPOS_START:ARM_QPOS_END], -np.pi, np.pi)

    return qpos[ARM_QPOS_START:ARM_QPOS_END].copy()


def main():
    # ---- Part A: Find the real fixed jaw fingertip position ----
    # Build model WITHOUT our custom fixed_finger, using only the
    # original collision meshes. Then position the arm at a pick pose
    # and measure where the wrist_roll_follower mesh extends to.
    print("=== Part A: Find real fixed jaw fingertip position ===")

    model_no_ff = build_model(with_fixed_finger=False)
    body_gripper = mujoco.mj_name2id(model_no_ff, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    body_mjaw = mujoco.mj_name2id(model_no_ff, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
    site_gf = mujoco.mj_name2id(model_no_ff, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    site_jc = mujoco.mj_name2id(model_no_ff, mujoco.mjtObj.mjOBJ_SITE, "jaw_contact")

    # List all geoms on gripper and moving_jaw bodies
    print("Gripper body geoms (original, no custom fixed_finger):")
    for i in range(model_no_ff.ngeom):
        bid = model_no_ff.geom_bodyid[i]
        if bid == body_gripper or bid == body_mjaw:
            name = mujoco.mj_id2name(model_no_ff, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
            gtype = model_no_ff.geom_type[i]
            gtype_name = {mujoco.mjtGeom.mjGEOM_BOX: "BOX", mujoco.mjtGeom.mjGEOM_MESH: "MESH"}
            tn = gtype_name.get(gtype, f"TYPE_{gtype}")
            sz = model_no_ff.geom_size[i]
            pos = model_no_ff.geom_pos[i]
            rbound = model_no_ff.geom_rbound[i]
            print(f"  geom {i}: '{name}' type={tn} pos_local={pos} size={sz} rbound={rbound:.4f}")
    print()

    # Position arm in a downward-reaching config with jaw CLOSED
    # and measure where the collision geoms are in world frame
    data = mujoco.MjData(model_no_ff)
    mujoco.mj_resetData(model_no_ff, data)

    # Try a config that reaches down: shoulder_lift more negative, elbow less bent
    # Home: [0.0, -0.5, 0.95, -0.55, 0.0]
    # Reaching down: [0.0, -1.2, 0.5, -0.8, 0.0]
    for config_name, arm_q in [
        ("home", [0.0, -0.5, 0.95, -0.55, 0.0]),
        ("reach_down_1", [0.0, -1.2, 0.5, -0.8, 0.0]),
        ("reach_down_2", [0.0, -1.0, 0.3, -1.0, 0.0]),
        ("reach_down_3", [0.0, -0.8, 0.6, -1.2, 0.0]),
    ]:
        data = mujoco.MjData(model_no_ff)
        mujoco.mj_resetData(model_no_ff, data)
        data.qpos[:5] = arm_q
        data.qpos[GRIPPER_QPOS_START] = -0.175  # jaw closed
        data.qpos[CUBE_QPOS_START:CUBE_QPOS_START+3] = [CUBE_XY[0], CUBE_XY[1], CUBE_Z_CENTER]
        data.qpos[CUBE_QPOS_START+3:CUBE_QPOS_START+7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model_no_ff, data)

        gf = data.site_xpos[site_gf].copy()
        jc = data.site_xpos[site_jc].copy() if site_jc >= 0 else np.zeros(3)

        print(f"\n  Config '{config_name}' (arm_q={arm_q}):")
        print(f"    gripperframe: {gf}")
        print(f"    jaw_contact:  {jc}")
        print(f"    gripper body: {data.xpos[body_gripper]}")
        print(f"    moving_jaw body: {data.xpos[body_mjaw]}")

        # Measure each geom's world position and bounding extent
        for i in range(model_no_ff.ngeom):
            bid = model_no_ff.geom_bodyid[i]
            if bid == body_gripper or bid == body_mjaw:
                name = mujoco.mj_id2name(model_no_ff, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
                gpos = data.geom_xpos[i].copy()
                gtype = model_no_ff.geom_type[i]
                if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                    sz = model_no_ff.geom_size[i]
                    # Box extends ±sz in each axis (in world frame after rotation)
                    # For simplicity, just report center z and half_z
                    print(f"    geom '{name}' world_pos={gpos}  (BOX size={sz})")
                elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
                    rbound = model_no_ff.geom_rbound[i]
                    print(f"    geom '{name}' world_pos={gpos}  (MESH rbound={rbound:.4f})")

    # ---- Part B: Orientation-constrained IK ----
    # Force the gripper to point downward, targeting gripperframe at various z
    print("\n\n=== Part B: Orientation-constrained IK (gripper pointing down) ===")

    # The gripper body's local axis that corresponds to the "approach direction"
    # (pointing toward fingertips). The gripperframe site has quat (0.707, 0, 0.707, 0)
    # in the gripper body, which rotates the local frame. The gripper body itself
    # has quat (0.0172, -0.0172, 0.7069, 0.7069) relative to wrist.
    #
    # For orientation constraint: we want the gripper's approach axis to point
    # in world -Z direction (straight down). Let's try constraining different
    # body axes and see which gives a downward-pointing gripper.

    # The moving_jaw body's local Z axis (0, 0, 1) corresponds to the jaw's
    # length axis. When the gripper points down, this axis should point
    # horizontally toward the cube (in world XY plane).
    # The gripper body's local -Y axis (0, -1, 0) might correspond to the
    # approach direction.

    # Let me try constraining the moving_jaw body's Z axis to point
    # in world -Z direction (downward). This should make the fingertips
    # point downward.

    body_mjaw_id = mujoco.mj_name2id(model_no_ff, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
    body_gripper_id = mujoco.mj_name2id(model_no_ff, mujoco.mjtObj.mjOBJ_BODY, "gripper")

    # Try different orientation constraints
    for ori_label, body_id, axis, world_dir in [
        ("mjaw_Z→-Z_down", body_mjaw_id, [0, 0, 1], [0, 0, -1]),
        ("mjaw_Y→-Z_down", body_mjaw_id, [0, 1, 0], [0, 0, -1]),
        ("gripper_Y→down_fwd", body_gripper_id, [0, -1, 0], [0.30, 0, -1]),  # approach toward cube
        ("gripper_Z→-Z_down", body_gripper_id, [0, 0, -1], [0, 0, -1]),
    ]:
        print(f"\n  Orientation: {ori_label}")
        print(f"  {'gf_tgt':>8}  {'gf_act':>8}  {'jc_z':>8}  {'lowest':>8}  {'safe?':>6}  {'cube_gap':>8}")

        for gf_z in [0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02]:
            data = mujoco.MjData(model_no_ff)
            mujoco.mj_resetData(model_no_ff, data)
            data.qpos[:5] = [0.0, -0.5, 0.95, -0.55, 0.0]
            data.qpos[GRIPPER_QPOS_START] = 0.5
            data.qpos[CUBE_QPOS_START:CUBE_QPOS_START+3] = [CUBE_XY[0], CUBE_XY[1], CUBE_Z_CENTER]
            data.qpos[CUBE_QPOS_START+3:CUBE_QPOS_START+7] = [1.0, 0.0, 0.0, 0.0]
            mujoco.mj_forward(model_no_ff, data)

            target_xyz = np.array([CUBE_XY[0], CUBE_XY[1], gf_z])
            seed = data.qpos.copy()
            seed[GRIPPER_QPOS_START] = 0.5

            arm_q = _ik_core(model_no_ff, site_gf, target_xyz, seed,
                body_id=body_id, target_axis=axis, target_world_dir=world_dir,
                ori_weight=2.0, pos_weight=1.0,
                step=0.5, damping=0.05, pos_tol=0.001, max_iters=200)

            data.qpos[ARM_QPOS_START:ARM_QPOS_END] = arm_q
            data.qpos[GRIPPER_QPOS_START] = 0.5
            data.qvel[:] = 0
            mujoco.mj_forward(model_no_ff, data)

            gf_actual = float(data.site_xpos[site_gf][2])
            jc_z = float(data.site_xpos[site_jc][2]) if site_jc >= 0 else -999

            # lowest gripper/moving_jaw point
            lowest_z = 999.0
            for i in range(model_no_ff.ngeom):
                bid = model_no_ff.geom_bodyid[i]
                if bid == body_gripper_id or bid == body_mjaw_id:
                    geom_z = float(data.geom_xpos[i][2])
                    gtype = model_no_ff.geom_type[i]
                    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                        half_z = float(model_no_ff.geom_size[i][2])
                        lowest_z = min(lowest_z, geom_z - half_z)
                    elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
                        rbound = float(model_no_ff.geom_rbound[i])
                        body_z = float(data.xpos[bid][2])
                        lowest_z = min(lowest_z, body_z - rbound)
                    else:
                        lowest_z = min(lowest_z, geom_z)

            cube_gap = jc_z - CUBE_Z_TOP if jc_z > -999 else -999
            safe = "OK" if lowest_z > 0 else "PEN"
            print(f"  {gf_z:8.4f}  {gf_actual:8.4f}  {jc_z:8.4f}  {lowest_z:8.4f}  {safe:>6}  {cube_gap:8.4f}")

    # ---- Part C: Find the correct fixed_finger position ----
    # With jaw CLOSED, measure where the moving jaw fingertip is
    # relative to the gripper body. The fixed finger should be at
    # the mirror position (opposite side of the jaw hinge).
    print("\n\n=== Part C: Find correct fixed_finger position ===")
    print("Measuring jaw_contact site position relative to gripper body")
    print("when arm is in various configurations with jaw CLOSED.")

    for config_name, arm_q in [
        ("home", [0.0, -0.5, 0.95, -0.55, 0.0]),
        ("reach_down", [0.0, -1.2, 0.5, -0.8, 0.0]),
    ]:
        data = mujoco.MjData(model_no_ff)
        mujoco.mj_resetData(model_no_ff, data)
        data.qpos[:5] = arm_q
        data.qpos[GRIPPER_QPOS_START] = -0.175  # jaw closed
        data.qpos[CUBE_QPOS_START:CUBE_QPOS_START+3] = [CUBE_XY[0], CUBE_XY[1], CUBE_Z_CENTER]
        data.qpos[CUBE_QPOS_START+3:CUBE_QPOS_START+7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model_no_ff, data)

        gf = data.site_xpos[site_gf].copy()
        jc = data.site_xpos[site_jc].copy() if site_jc >= 0 else np.zeros(3)
        gripper_pos = data.xpos[body_gripper_id].copy()
        gripper_rot = data.xmat[body_gripper_id].reshape(3, 3).copy()

        # jaw_contact in gripper body local frame
        jc_local = gripper_rot.T @ (jc - gripper_pos)

        print(f"\n  Config '{config_name}':")
        print(f"    gripper world pos: {gripper_pos}")
        print(f"    jaw_contact world: {jc}")
        print(f"    jaw_contact in gripper local frame: {jc_local}")
        print(f"    gripperframe world: {gf}")
        print(f"    gripperframe in gripper local frame: {gripper_rot.T @ (gf - gripper_pos)}")

        # The moving jaw body position in gripper local frame is (0.0202, 0.0188, -0.0234)
        # When jaw is closed (θ=-0.175), the jaw_contact site should be near the fixed jaw
        # The fixed jaw fingertip should be at a similar position but on the opposite Y side

        # Moving jaw body in gripper local frame
        mjaw_pos = data.xpos[body_mjaw_id].copy()
        mjaw_local = gripper_rot.T @ (mjaw_pos - gripper_pos)
        print(f"    moving_jaw body in gripper local: {mjaw_local}")

        # When jaw is nearly closed, the fingertip should be at:
        # jaw_origin + rotation(-0.175) * jaw_local_fingertip_offset
        # The fixed jaw fingertip should be at a similar position
        # but on the opposite side of the X axis (since the jaw rotates
        # toward the fixed jaw)

        # Let's compute what position in gripper local frame corresponds
        # to "the fixed jaw fingertip" — it should be at a similar z
        # as jc_local but on the opposite side

        # The fixed jaw is the wrist_roll_follower mesh. Let's find
        # where that mesh's bounding box lowest point is
        for i in range(model_no_ff.ngeom):
            bid = model_no_ff.geom_bodyid[i]
            if bid == body_gripper_id:
                name = mujoco.mj_id2name(model_no_ff, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
                if model_no_ff.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
                    gpos_world = data.geom_xpos[i].copy()
                    gpos_local = gripper_rot.T @ (gpos_world - gripper_pos)
                    rbound = model_no_ff.geom_rbound[i]
                    print(f"    mesh geom '{name}' world_pos={gpos_world}  local_pos={gpos_local}  rbound={rbound:.4f}")


if __name__ == "__main__":
    main()
