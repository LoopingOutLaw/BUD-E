"""Inverse Kinematics controller for SO-101 arm using MuJoCo's Jacobian.

Copied from ggand0/pick-101 src/controllers/ik_controller.py — the proven
working IK implementation for SO-101 grasping. Uses damped least-squares
with optional locked joints and orientation constraint.
"""
import mujoco
import numpy as np


class IKController:
    """Damped least-squares IK controller.

    Uses MuJoCo's mj_jac to compute the Jacobian and solve for joint velocities
    that move the end-effector toward a target position.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        end_effector_site: str = "gripperframe",
        damping: float = 0.1,
        max_dq: float = 0.5,
    ):
        self.model = model
        self.data = data
        self.damping = damping
        self.max_dq = max_dq

        self.ee_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, end_effector_site
        )
        if self.ee_site_id == -1:
            raise ValueError(f"Site '{end_effector_site}' not found in model")

        self.n_arm_joints = 5
        self.n_total_joints = model.nv
        self.jacp = np.zeros((3, self.n_total_joints))
        self.jacr = np.zeros((3, self.n_total_joints))

    def get_ee_position(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_ee_orientation(self) -> np.ndarray:
        return self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

    def _quat_to_mat(self, quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])

    def _orientation_error(self, target_mat: np.ndarray, current_mat: np.ndarray) -> np.ndarray:
        R_error = target_mat @ current_mat.T
        trace = np.trace(R_error)
        cos_theta = np.clip((trace - 1) / 2, -1, 1)
        theta = np.arccos(cos_theta)

        if theta < 1e-6:
            return np.zeros(3)

        axis = np.array([
            R_error[2, 1] - R_error[1, 2],
            R_error[0, 2] - R_error[2, 0],
            R_error[1, 0] - R_error[0, 1]
        ]) / (2 * np.sin(theta))

        return axis * theta

    def compute_joint_velocities(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None = None,
        orientation_weight: float = 1.0,
        locked_joints: list[int] | None = None,
    ) -> np.ndarray:
        current_pos = self.get_ee_position()
        pos_error = target_pos - current_pos

        mujoco.mj_jacSite(
            self.model, self.data,
            self.jacp, self.jacr,
            self.ee_site_id
        )

        if locked_joints is None:
            active_joints = list(range(self.n_arm_joints))
        else:
            active_joints = [i for i in range(self.n_arm_joints) if i not in locked_joints]

        n_active = len(active_joints)

        Jp = self.jacp[:, active_joints]
        Jr = self.jacr[:, active_joints]

        if target_quat is not None:
            target_mat = self._quat_to_mat(target_quat)
            current_mat = self.get_ee_orientation()
            ori_error = self._orientation_error(target_mat, current_mat) * orientation_weight
            error = np.concatenate([pos_error, ori_error])
            J = np.vstack([Jp, Jr])
        else:
            error = pos_error
            J = Jp

        JTJ = J.T @ J
        damping_matrix = self.damping**2 * np.eye(n_active)

        try:
            dq_active = np.linalg.solve(JTJ + damping_matrix, J.T @ error)
        except np.linalg.LinAlgError:
            dq_active = np.linalg.pinv(J) @ error

        dq_active = np.clip(dq_active, -self.max_dq, self.max_dq)

        dq = np.zeros(self.n_arm_joints)
        for i, joint_idx in enumerate(active_joints):
            dq[joint_idx] = dq_active[i]

        return dq

    def step_toward_target(
        self,
        target_pos: np.ndarray,
        gripper_action: float = 0.0,
        gain: float = 1.0,
        target_quat: np.ndarray | None = None,
        orientation_weight: float = 1.0,
        locked_joints: list[int] | None = None,
    ) -> np.ndarray:
        """Compute control signal to move toward target position.

        Returns full control vector (6,) for all actuators.
        Gripper action maps -1..1 to actuator control range.
        """
        dq = self.compute_joint_velocities(target_pos, target_quat, orientation_weight, locked_joints)
        dq *= gain

        current_q = self.data.qpos[:self.n_arm_joints].copy()
        target_q = current_q + dq

        for i in range(self.n_arm_joints):
            jnt_range = self.model.jnt_range[i]
            if jnt_range[0] != jnt_range[1]:
                target_q[i] = np.clip(target_q[i], jnt_range[0], jnt_range[1])

        ctrl = np.zeros(self.model.nu)
        ctrl[:self.n_arm_joints] = target_q

        gripper_range = self.model.actuator_ctrlrange[5]
        gripper_ctrl = (gripper_action + 1) / 2 * (gripper_range[1] - gripper_range[0]) + gripper_range[0]
        ctrl[5] = gripper_ctrl

        return ctrl
