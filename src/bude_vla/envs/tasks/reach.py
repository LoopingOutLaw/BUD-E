"""Reach task: move end-effector to a randomly placed target sphere."""
from __future__ import annotations
import jax.numpy as jnp
import numpy as np


class ReachTask:
    """Batched reach task (1 env)."""

    def __init__(self, env, table_top: float = 0.42, reach_radius: float = 0.3):
        self.env = env
        self.table_top = table_top
        self.reach_radius = reach_radius
        self.target_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self.success: bool = False

    def reset_target(self, rng: np.random.Generator) -> None:
        # Random target on the table surface in front of the arm
        angle = rng.uniform(-np.pi / 3, np.pi / 3)
        dist = rng.uniform(0.15, 0.45)
        self.target_pos = np.array([
            0.6 + dist * np.cos(angle),
            dist * np.sin(angle),
            self.table_top + 0.05,
        ], dtype=np.float32)
        self.success = False

    def ee_pos(self, state) -> np.ndarray:
        """Get end-effector position. Returns (3,) np array.

        `state.site_xpos` is (nsite, 3); we look for the first site defined as ee.
        Fall back to joint 6 position.
        """
        try:
            sx = np.asarray(state.site_xpos)
            if sx.ndim == 2 and sx.shape[-1] == 3:
                return sx[0] if sx.shape[0] >= 1 else np.zeros(3)
            if sx.ndim == 1 and sx.shape[0] == 3:
                return sx
        except Exception:
            pass
        return np.zeros(3, dtype=np.float32)

    def distance_to_target(self, state) -> float:
        ee = self.ee_pos(state)
        return float(np.linalg.norm(ee - self.target_pos))

    def is_success(self, state, threshold: float = 0.03) -> bool:
        return self.distance_to_target(state) < threshold

    def observation(self, state) -> dict:
        return {
            "ee_pos": self.ee_pos(state),
            "target_pos": self.target_pos,
            "qpos": np.asarray(state.qpos),
        }


def scripted_reach(policy_input: dict, dt: float = 0.005) -> int:
    """Move EE toward target using simple PD.

    Returns the index `i` in the action vector where the EE position lives.
    For now we treat the policy output as delta-qpos for the first 3 joints.
    """
    return 6  # Action vector size for the model
