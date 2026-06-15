"""Tests for bude_vla.ik inverse-kinematics solver."""
import mujoco
import numpy as np
from pathlib import Path
from bude_vla.ik import solve_ik_to_xyz


MODEL_PATH = Path(__file__).resolve().parents[2] / "urdf" / "ur5e_scene.xml"


def test_solve_ik_moves_ee_toward_target():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_center")
    ee_xyz_before = data.site_xpos[site_id].copy()

    target = ee_xyz_before + np.array([0.05, 0.0, 0.0])
    new_arm_qpos = solve_ik_to_xyz(model, data, target, data.qpos.copy())

    data.qpos[7:13] = new_arm_qpos
    mujoco.mj_forward(model, data)
    ee_xyz_after = data.site_xpos[site_id].copy()

    d_before = np.linalg.norm(target - ee_xyz_before)
    d_after = np.linalg.norm(target - ee_xyz_after)
    assert d_after < d_before, f"EE did not move toward target: {d_before=:.4f} {d_after=:.4f}"


def test_solve_ik_converges_within_tolerance():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_center")
    ee_xyz = data.site_xpos[site_id].copy()
    target = ee_xyz + np.array([0.03, -0.02, 0.01])

    new_arm_qpos = solve_ik_to_xyz(model, data, target, data.qpos.copy(),
                                     pos_tol=0.01, max_step=0.05, max_iters=50)
    data.qpos[7:13] = new_arm_qpos
    mujoco.mj_forward(model, data)
    ee_final = data.site_xpos[site_id].copy()

    dist = np.linalg.norm(target - ee_final)
    assert dist < 0.03, f"EE did not converge: dist={dist:.4f}"
