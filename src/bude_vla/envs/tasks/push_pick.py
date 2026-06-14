"""Push task: push cube to target zone on the table."""
from __future__ import annotations
import numpy as np


class PushTask:
    """Batched push task. The cube starts at a random pose, target is a plane.

    This is a placeholder — full scripted policy comes when we add cube geoms
    to the MJCF in a follow-up task. For now this provides the interface.
    """

    def __init__(self, env, table_top: float = 0.42):
        self.env = env
        self.table_top = table_top
        self.cube_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self.target_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self.success: bool = False

    def reset_episode(self, rng: np.random.Generator) -> None:
        # Cube placed near the arm
        self.cube_pos = np.array([
            0.6 + rng.uniform(-0.1, 0.1),
            rng.uniform(-0.2, 0.2),
            self.table_top + 0.025,
        ], dtype=np.float32)
        # Target zone somewhere on the table
        self.target_pos = np.array([
            0.6 + rng.uniform(-0.25, 0.25),
            rng.uniform(-0.3, 0.3),
            self.table_top + 0.025,
        ], dtype=np.float32)
        self.success = False

    def is_success(self, *, threshold: float = 0.04) -> bool:
        return bool(np.linalg.norm(self.cube_pos - self.target_pos) < threshold)


class PickPlaceTask:
    """Batched pick-and-place task."""

    def __init__(self, env, table_top: float = 0.42):
        self.env = env
        self.table_top = table_top
        self.cube_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self.target_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self.cube_in_gripper: bool = False
        self.success: bool = False

    def reset_episode(self, rng: np.random.Generator) -> None:
        self.cube_pos = np.array([
            0.6 + rng.uniform(-0.1, 0.1),
            rng.uniform(-0.15, 0.15),
            self.table_top + 0.025,
        ], dtype=np.float32)
        self.target_pos = np.array([
            0.6 + rng.uniform(-0.2, 0.2),
            rng.uniform(-0.25, 0.25),
            self.table_top + 0.025,
        ], dtype=np.float32)
        self.cube_in_gripper = False
        self.success = False

    def is_success(self, *, threshold: float = 0.05) -> bool:
        if not self.cube_in_gripper:
            return False
        return bool(np.linalg.norm(self.cube_pos - self.target_pos) < threshold)
