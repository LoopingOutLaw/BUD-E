"""Inverse-kinematics via Jacobian transpose for UR5e-style 6-DOF arm."""
from __future__ import annotations
import numpy as np
import mujoco


def solve_ik_to_xyz(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_xyz: np.ndarray,
    current_qpos: np.ndarray,
    site_name: str = "ee_center",
    max_step: float = 0.05,
    pos_tol: float = 0.005,
    max_iters: int = 20,
) -> np.ndarray:
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    qpos = current_qpos.copy()
    data_copy = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

    for _ in range(max_iters):
        data_copy.qpos[:] = qpos
        mujoco.mj_forward(model, data_copy)

        ee_xyz = data_copy.site_xpos[site_id]
        err = target_xyz - ee_xyz
        if np.linalg.norm(err) < pos_tol:
            break

        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data_copy, jacp, jacr, site_id)

        dq = max_step * (jacp.T @ err)
        qpos[7:13] += dq[6:12]
        qpos[7:13] = np.clip(qpos[7:13], -np.pi, np.pi)

    return qpos[7:13].copy()
