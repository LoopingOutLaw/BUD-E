"""Inverse-kinematics solvers for SO-101 5-DOF arm via MuJoCo.

Two solvers provided:
- solve_ik_to_xyz_dls: damped least-squares (Levenberg-Marquardt-like)
- solve_ik_to_xyz: faster Jacobian-transpose (kept for backwards compat tests)

Orientation-constrained IK added for GRASP phase: with 5 DOF we can control
3 position + 2 orientation. This prevents the gripper from drifting into
sideways orientations that push the ball out of the bowl.
"""
from __future__ import annotations
import numpy as np
import mujoco

from bude_vla.envs.so101_mjx import ARM_QPOS_START, ARM_QPOS_END


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
             # ---- orientation constraint (optional) ----
             body_id: int | None = None,
             target_axis: np.ndarray | None = None,
             target_world_dir: np.ndarray | None = None,
             ori_weight: float = 2.0,
             pos_weight: float = 1.0,
             ori_tol: float = 0.05,
             ) -> np.ndarray:
    """Damped least-squares IK with optional orientation constraint.

    Position-only IK on a 5-DOF arm has a 2-D nullspace.  The solver
    naturally drifts into whatever orientation is closest to the seed
    pose -- which is often a side-on glancing pose that pushes the ball
    sideways out of the bowl rather than pinching it.

    With an orientation constraint we solve 3 position + 2 orientation
    = 5 constraints, exactly matching the 5 arm DOF.  We constrain one
    body axis (e.g. the jaw's +Z length axis) to point in a desired
    world direction (e.g. horizontally toward the ball).
    """
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    qpos = current_qpos.copy()
    data_copy = mujoco.MjData(model)

    use_ori = (body_id is not None
               and target_axis is not None
               and target_world_dir is not None)
    target_axis_body = None
    target_world_dir_unit = None
    if use_ori:
        target_axis_body = np.asarray(target_axis, dtype=np.float64)
        target_axis_body = target_axis_body / max(np.linalg.norm(target_axis_body), 1e-12)
        target_world_dir_unit = np.asarray(target_world_dir, dtype=np.float64)
        norm = np.linalg.norm(target_world_dir_unit)
        if norm > 1e-12:
            target_world_dir_unit = target_world_dir_unit / norm
        else:
            target_world_dir_unit = np.array([1.0, 0.0, 0.0])

    for _ in range(max_iters):
        data_copy.qpos[:] = qpos
        mujoco.mj_forward(model, data_copy)

        ee_xyz = data_copy.site_xpos[site_id]
        err_pos = target_xyz - ee_xyz
        pos_err_norm = np.linalg.norm(err_pos)

        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data_copy, jacp, jacr, site_id)

        J_pos = jacp[:, ARM_QPOS_START:ARM_QPOS_END]

        if use_ori:
            R = data_copy.xmat[body_id].reshape(3, 3)
            current_world_dir = R @ target_axis_body
            # Error = sin(theta) * axis_perp, equivalent to -(target x current)
            # for small angles. Use cross-product direction (target_world_dir x current).
            err_ori = np.cross(target_world_dir_unit, current_world_dir)
            ori_err_norm = np.linalg.norm(err_ori)

            J_ori = jacr[:, ARM_QPOS_START:ARM_QPOS_END]

            err = np.concatenate([err_pos * pos_weight, err_ori * ori_weight])
            J = np.vstack([J_pos * pos_weight, J_ori * ori_weight])
        else:
            ori_err_norm = 0.0
            err = err_pos
            J = J_pos

        if pos_err_norm < pos_tol and ori_err_norm < ori_tol:
            break

        if method == "dls":
            JJt = J @ J.T
            dq_arm = step * J.T @ np.linalg.solve(
                JJt + (damping ** 2) * np.eye(JJt.shape[0]), err
            )
        elif method == "jt":
            dq = step * (jacp.T @ err_pos)
            dq_arm = dq[ARM_QPOS_START:ARM_QPOS_END]
        else:
            raise ValueError(f"Unknown method: {method}")

        qpos[ARM_QPOS_START:ARM_QPOS_END] += dq_arm
        qpos[ARM_QPOS_START:ARM_QPOS_END] = np.clip(
            qpos[ARM_QPOS_START:ARM_QPOS_END], -np.pi, np.pi
        )

    return qpos[ARM_QPOS_START:ARM_QPOS_END].copy()


def solve_ik_to_xyz(model: mujoco.MjModel,
                    data: mujoco.MjData,
                    target_xyz: np.ndarray,
                    current_qpos: np.ndarray,
                    site_name: str = "gripperframe",
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
                        site_name: str = "gripperframe",
                        **kwargs) -> np.ndarray:
    """Damped least-squares IK. Robust near singularities, faster convergence."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    kwargs.setdefault("method", "dls")
    kwargs.setdefault("step", 0.5)
    kwargs.setdefault("damping", 0.05)
    kwargs.setdefault("max_iters", 30)
    kwargs.setdefault("pos_tol", 0.005)
    return _ik_core(model, site_id, target_xyz, current_qpos, **kwargs)
