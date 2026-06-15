"""Inverse-kinematics solvers for UR5e-style 6-DOF arm via MuJoCo.

Two solvers provided:
- solve_ik_to_xyz_dls: damped least-squares (Levenberg-Marquardt-like)
- solve_ik_to_xyz: faster Jacobian-transpose (kept for backwards compat tests)
"""
from __future__ import annotations
import numpy as np
import mujoco


def _ik_core(model: mujoco.MjModel,
             site_id: int,
             target_xyz: np.ndarray,
             current_qpos: np.ndarray,
             *,
             method: str = "dls",
             step: float = 0.5,
             damping: float = 0.05,
             pos_tol: float = 0.005,
             max_iters: int = 50,
             ) -> np.ndarray:
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    qpos = current_qpos.copy()
    data_copy = mujoco.MjData(model)

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

        if method == "dls":
            J = jacp[:, 6:12]
            JJt = J @ J.T
            dq_arm = step * J.T @ np.linalg.solve(JJt + (damping ** 2) * np.eye(3), err)
        elif method == "jt":
            dq = step * (jacp.T @ err)
            dq_arm = dq[6:12]
        else:
            raise ValueError(f"Unknown method: {method}")

        qpos[7:13] += dq_arm
        qpos[7:13] = np.clip(qpos[7:13], -np.pi, np.pi)

    return qpos[7:13].copy()


def solve_ik_to_xyz(model: mujoco.MjModel,
                    data: mujoco.MjData,
                    target_xyz: np.ndarray,
                    current_qpos: np.ndarray,
                    site_name: str = "ee_center",
                    **kwargs) -> np.ndarray:
    """Backwards-compat: Jacobian-transpose IK."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    kwargs.setdefault("method", "jt")
    kwargs.setdefault("step", 0.05)
    kwargs.setdefault("max_iters", 20)
    kwargs.setdefault("pos_tol", 0.005)
    return _ik_core(model, site_id, target_xyz, current_qpos, **kwargs)


def solve_ik_to_xyz_dls(model: mujoco.MjModel,
                        data: mujoco.MjData,
                        target_xyz: np.ndarray,
                        current_qpos: np.ndarray,
                        site_name: str = "ee_center",
                        **kwargs) -> np.ndarray:
    """Damped least-squares IK. Robust near singularities, faster convergence."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    kwargs.setdefault("method", "dls")
    kwargs.setdefault("step", 0.5)
    kwargs.setdefault("damping", 0.05)
    kwargs.setdefault("max_iters", 30)
    kwargs.setdefault("pos_tol", 0.005)
    return _ik_core(model, site_id, target_xyz, current_qpos, **kwargs)
