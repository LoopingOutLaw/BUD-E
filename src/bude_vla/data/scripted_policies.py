"""Push scripted policy: lower gripper toward cube, push toward target zone.

The cube starts at (~0.6, 0, 0.445) with some random y-offset. Target zone is at
(0.85, 0, 0.421). Policy:
  - Phase 1 (approach): drive EE toward (cube.x, cube.y - 0.05) for ~10 steps
  - Phase 2 (push): hold EE at cube.y and ride the cube in +x until z target reached
"""
from __future__ import annotations
import numpy as np


def scripted_push_step(ee: np.ndarray, cube: np.ndarray, target: np.ndarray,
                        phase: int, nu: int = 7) -> tuple[np.ndarray, int]:
    """One push step.

    Returns (action, next_phase).
    phase==0: approach (above+behind cube)
    phase==1: push (touching cube, ride it forward)
    """
    action = np.zeros(nu, dtype=np.float32)
    if nu < 3:
        return action, phase

    if phase == 0:
        # Approach: get behind cube (small -y offset) and just above
        goal = np.array([cube[0] - 0.08, cube[1] - 0.05, cube[2] + 0.02], dtype=np.float32)
        delta = goal - ee
        action[0] = np.clip(-delta[1] * 4.0, -1, 1)
        action[1] = np.clip(-delta[0] * 5.0, -1, 1)
        action[2] = np.clip(delta[2] * 3.0, -1, 1)
        # Switch to push when EE is close enough to the ride pose
        if np.linalg.norm(delta[:2]) < 0.04:
            return action, 1
        return action, 0
    else:
        # Push: hold EE above+behind cube, push +x target
        push_target = np.array([cube[0] + 0.05, target[1], cube[2] + 0.02],
                                dtype=np.float32)
        delta = push_target - ee
        action[0] = np.clip(-delta[1] * 4.0, -1, 1)
        action[1] = np.clip(cube[1] - ee[1], -1, 1) * 8.0
        action[1] = np.clip(action[1], -1, 1)
        action[2] = np.clip(delta[2] * 3.0, -1, 1)
        return action, 1
