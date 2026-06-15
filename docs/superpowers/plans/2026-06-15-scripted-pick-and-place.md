# Scripted Pick-and-Place Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a scripted pick-and-place policy that drives the UR5e arm to grip a cube, lift it, carry it to a blue target zone, and release it. Record 100 clean demos, train BUD-E, render a single eval-rollout MP4.

**Architecture:** Jacobian-transpose IK solver drives EE to target poses via a 6-phase state machine (approach → descend → grip → lift → move → release). Pure CPython/MuJoCo (no JAX) for recording; MJX for batched eval after training. Each recording episode randomizes cube (x,y) on the table.

**Tech Stack:** MuJoCo (C API via pymujoco), numpy, torch (training), imageio+ffmpeg (video), pytest (TDD)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/bude_vla/ik.py` | Jacobian-transpose IK solver: `solve_ik_to_xyz()` |
| `src/bude_vla/scripted_pick_and_place.py` | `ScriptedPickAndPlace` state machine, 6 phases |
| `src/bude_vla/data/pick_recorder.py` | Record pick-and-place episodes to LeRobot v3 layout |
| `scripts/record_pick_episodes.py` | CLI: runs `pick_recorder`, saves to `data/pick_v3/` |
| `scripts/train.py` | Modified: add `--task pick` path |
| `scripts/render_pick_rollout.py` | Load checkpoint, run 1 rollout, render MP4 |
| `tests/ik/test_ik.py` | TDD tests for IK solver |
| `tests/test_scripted_pick_and_place.py` | TDD tests for state machine |

---

### Task 1: Jacobian-transpose IK solver

**Files:**
- Create: `src/bude_vla/ik.py`
- Test: `tests/ik/test_ik.py`

- [ ] **Step 1: Write failing test — IK moves EE toward a target**

```python
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

    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_center")
    ee_xyz_before = data.site_xpos[ee_site_id].copy()

    target = ee_xyz_before + np.array([0.05, 0.0, 0.0])
    new_arm_qpos = solve_ik_to_xyz(model, data, target, data.qpos.copy())

    data.qpos[7:13] = new_arm_qpos
    mujoco.mj_forward(model, data)
    ee_xyz_after = data.site_xpos[ee_site_id].copy()

    d_before = np.linalg.norm(target - ee_xyz_before)
    d_after = np.linalg.norm(target - ee_xyz_after)
    assert d_after < d_before, f"EE did not move toward target: {d_before=:.4f} {d_after=:.4f}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aditya/bude_vla && unset PYTHONPATH && PYTHONPATH=/home/aditya/bude_vla/src timeout 30 /home/aditya/.bude-venv/bin/python -m pytest tests/ik/test_ik.py -v --tb=short`
Expected: FAIL with `ImportError: cannot import name 'solve_ik_to_xyz'`

- [ ] **Step 3: Write minimal implementation**

```python
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
    """Return new arm qpos[7:13] that moves EE toward target_xyz.

    Uses Jacobian-transpose IK: dq = alpha * J^T * e.
    Does NOT call mj_step — caller decides when to integrate.
    """
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
        arm_dof = dq[6:12]
        qpos[7:13] += arm_dof
        qpos[7:13] = np.clip(qpos[7:13], -np.pi, np.pi)

    return qpos[7:13].copy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aditya/bude_vla && unset PYTHONPATH && PYTHONPATH=/home/aditya/bude_vla/src timeout 30 /home/aditya/.bude-venv/bin/python -m pytest tests/ik/test_ik.py -v --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/aditya/bude_vla && git add src/bude_vla/ik.py tests/ik/test_ik.py && git commit -m "feat(ik): Jacobian-transpose IK solver for UR5e EE site"
```

---

### Task 2: Scripted pick-and-place state machine

**Files:**
- Create: `src/bude_vla/scripted_pick_and_place.py`
- Test: `tests/test_scripted_pick_and_place.py`

- [ ] **Step 1: Write failing test — policy reaches cube at known pose**

```python
"""Tests for scripted pick-and-place policy."""
import mujico
import numpy as np
from pathlib import Path
from bude_vla.scripted_pick_and_place import ScriptedPickAndPlace


MODEL_PATH = Path(__file__).resolve().parents[1] / "urdf" / "ur5e_scene.xml"


def test_policy_approaches_cube_at_known_pose():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    cube_xy = np.array([0.6, 0.0])
    policy = ScriptedPickAndPlace(model, data, cube_start_xy=cube_xy)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_center")
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")

    for _ in range(80):
        ctrl, done, info = policy.step(model, data)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        if done:
            break

    mujoco.mj_forward(model, data)
    ee_xyz = data.site_xpos[site_id]
    cube_xyz = data.xpos[cube_body_id]
    dist = np.linalg.norm(ee_xyz - cube_xyz)
    assert dist < 0.20, f"EE did not approach cube: dist={dist:.3f}"


def test_policy_advances_phases():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    policy = ScriptedPickAndPlace(model, data, cube_start_xy=np.array([0.6, 0.0]))
    phases_seen = set()
    for _ in range(250):
        ctrl, done, info = policy.step(model, data)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        phases_seen.add(info["phase"])
        if done:
            break

    assert 0 in phases_seen, "Never entered APPROACH phase"
    assert len(phases_seen) > 1, f"Policy never advanced past phase 0, saw {phases_seen}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aditya/bude_vla && unset PYTHONPATH && PYTHONPATH=/home/aditya/bude_vla/src timeout 30 /home/aditya/.bude-venv/bin/python -m pytest tests/test_scripted_pick_and_place.py -v --tb=short`
Expected: FAIL with `ImportError: cannot import name 'ScriptedPickAndPlace'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Scripted pick-and-place policy: approach → descend → grip → lift → move → release."""
from __future__ import annotations
import numpy as np
import mujoco
from bude_vla.ik import solve_ik_to_xyz


APPROACH = 0
DESCEND = 1
GRIP = 2
LIFT = 3
MOVE = 4
RELEASE = 5
TABLE_Z = 0.42
CUBE_HALF = 0.025
HOVER_ABOVE_CUBE = 0.15


class ScriptedPickAndPlace:
    def __init__(self, model, data, cube_start_xy, target_xy=(0.85, 0.0)):
        self.model = model
        self.cube_start_xy = np.asarray(cube_start_xy, dtype=np.float64)
        self.target_xy = np.asarray(target_xy, dtype=np.float64)
        self.phase = APPROACH
        self.phase_step = 0
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_center")
        self.cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._max_steps = 300
        self._total_steps = 0

    def _ee_xyz(self, data):
        return data.site_xpos[self.site_id].copy()

    def _cube_xyz(self, data):
        return data.xpos[self.cube_body_id].copy()

    def _ik_ctrl(self, data, target_xyz):
        """Compute arm ctrl to move EE toward target_xyz via IK, then PD to track."""
        new_arm = solve_ik_to_xyz(self.model, data, target_xyz, data.qpos.copy(), max_step=0.03, pos_tol=0.008, max_iters=10)
        ctrl = np.zeros(7, dtype=np.float32)
        for i in range(6):
            err = new_arm[i] - data.qpos[7 + i]
            ctrl[i] = np.clip(err * 8.0, -1.0, 1.0)
        return ctrl

    def step(self, model, data):
        self._total_steps += 1
        self.phase_step += 1
        ctrl = np.zeros(7, dtype=np.float32)
        done = False
        cube = self._cube_xyz(data)
        ee = self._ee_xyz(data)

        if self.phase == APPROACH:
            goal = np.array([self.cube_start_xy[0] - 0.05, self.cube_start_xy[1], TABLE_Z + CUBE_HALF + HOVER_ABOVE_CUBE])
            ctrl = self._ik_ctrl(data, goal)
            dist = np.linalg.norm(ee - goal)
            if dist < 0.04 or self.phase_step > 60:
                self.phase = DESCEND
                self.phase_step = 0

        elif self.phase == DESCEND:
            goal = np.array([self.cube_start_xy[0] - 0.02, self.cube_start_xy[1], TABLE_Z + CUBE_HALF + 0.04])
            ctrl = self._ik_ctrl(data, goal)
            dist = np.linalg.norm(ee - goal)
            if dist < 0.04 or self.phase_step > 50:
                self.phase = GRIP
                self.phase_step = 0

        elif self.phase == GRIP:
            ctrl = self._ik_ctrl(data, np.array([cube[0], cube[1], cube[2] + 0.03]))
            ctrl[6] = 1.0
            if self.phase_step > 40:
                self.phase = LIFT
                self.phase_step = 0

        elif self.phase == LIFT:
            goal = np.array([cube[0], cube[1], TABLE_Z + 0.30])
            ctrl = self._ik_ctrl(data, goal)
            ctrl[6] = 1.0
            dist = np.linalg.norm(ee - goal)
            if dist < 0.06 or self.phase_step > 50:
                self.phase = MOVE
                self.phase_step = 0

        elif self.phase == MOVE:
            goal = np.array([self.target_xy[0], self.target_xy[1], TABLE_Z + 0.25])
            ctrl = self._ik_ctrl(data, goal)
            ctrl[6] = 1.0
            dist = np.linalg.norm(ee - goal)
            if dist < 0.06 or self.phase_step > 60:
                self.phase = RELEASE
                self.phase_step = 0

        elif self.phase == RELEASE:
            ctrl = self._ik_ctrl(data, np.array([self.target_xy[0], self.target_xy[1], TABLE_Z + CUBE_HALF + 0.05]))
            ctrl[6] = -1.0
            if self.phase_step > 30:
                done = True

        if self._total_steps >= self._max_steps:
            done = True

        return ctrl, done, {"phase": self.phase, "phase_step": self.phase_step}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aditya/bude_vla && unset PYTHONPATH && PYTHONPATH=/home/aditya/bude_vla/src timeout 30 /home/aditya/.bude-venv/bin/python -m pytest tests/test_scripted_pick_and_place.py -v --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/aditya/bude_vla && git add src/bude_vla/scripted_pick_and_place.py src/bude_vla/ik.py tests/test_scripted_pick_and_place.py && git commit -m "feat: scripted pick-and-place policy with 6-phase state machine"
```

---

### Task 3: Pick-and-place episode recorder

**Files:**
- Create: `src/bude_vla/data/pick_recorder.py`
- Create: `scripts/record_pick_episodes.py`
- Test: `tests/data/test_pick_recorder.py`

- [ ] **Step 1: Write failing test — recorder writes one episode**

```python
"""Tests for pick_recorder."""
import tempfile
from pathlib import Path
from bude_vla.data.pick_recorder import record_pick_episode


def test_record_pick_episode_writes_files():
    with tempfile.TemporaryDirectory() as td:
        ep = record_pick_episode(root=td, episode_idx=0, cube_xy=(0.6, 0.0))
        assert "images" in ep
        assert "proprio" in ep
        assert "actions" in ep
        assert "instruction" in ep
        assert ep["images"].shape[1:] == (64, 64, 3)
        assert ep["proprio"].shape[1] == 8
        assert ep["actions"].shape[1] == 7
        assert "pick" in ep["instruction"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aditya/bude_vla && unset PYTHONPATH && PYTHONPATH=/home/aditya/bude_vla/src timeout 30 /home/aditya/.bude-venv/bin/python -m pytest tests/data/test_pick_recorder.py -v --tb=short`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

`pick_recorder.py` mirrors `cpu_recorder.py` but uses `ScriptedPickAndPlace`:

```python
"""Record pick-and-place episodes for BUD-E training."""
from __future__ import annotations
import mujoco
import numpy as np
from pathlib import Path
from bude_vla.scripted_pick_and_place import ScriptedPickAndPlace

INSTRUCTION = "pick up the red cube and place it in the blue target zone"


def record_pick_episode(root: str | Path, episode_idx: int = 0,
                        cube_xy: tuple[float, float] = (0.6, 0.0),
                        camera: str = "front_top", img_size: int = 64,
                        max_steps: int = 300) -> dict:
    from bude_vla.envs.so101_mjx import ARM_MODEL_PATH
    model = mujoco.MjModel.from_xml_path(str(ARM_MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    data.qpos[0:3] = [cube_xy[0], cube_xy[1], 0.445]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=img_size, width=img_size)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    renderer.scene.camera(mujoco.MjvCamera())

    policy = ScriptedPickAndPlace(model, data, cube_start_xy=np.array(cube_xy))

    images, proprios, actions = [], [], []
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")

    for _ in range(max_steps):
        renderer.update_scene(data, camera_id=cam_id)
        img = renderer.render()
        images.append(img)
        proprios.append(data.qpos[7:15].astype(np.float32).copy())

        ctrl, done, info = policy.step(model, data)
        actions.append(ctrl.copy())
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if done:
            break

    renderer.close()
    cube_final = data.xpos[cube_body_id].copy()
    target_pos = data.xpos[target_body_id].copy()
    success = np.linalg.norm(cube_final[:2] - target_pos[:2]) < 0.08

    return {
        "images": np.array(images, dtype=np.uint8),
        "proprio": np.array(proprios, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "instruction": INSTRUCTION,
        "success": bool(success),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aditya/bude_vla && unset PYTHONPATH && PYTHONPATH=/home/aditya/bude_vla/src timeout 60 /home/aditya/.bude-venv/bin/python -m pytest tests/data/test_pick_recorder.py -v --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/aditya/bude_vla && git add src/bude_vla/data/pick_recorder.py tests/data/test_pick_recorder.py && git commit -m "feat(data): pick-and-place episode recorder with success detection"
```

---

### Task 4: Batch recording script (record 100 episodes)

**Files:**
- Create: `scripts/record_pick_episodes.py`

- [ ] **Step 1: Write recording script**

```python
"""Record 100 pick-and-place episodes with randomized cube positions.

Usage: PYTHONPATH=src python scripts/record_pick_episodes.py
"""
import numpy as np
from bude_vla.data.pick_recorder import record_pick_episode
from bude_vla.data.lerobot_v3 import write_episode

ROOT = "/home/aditya/bude_vla/data/pick_v3"
N_EPISODES = 100
CUBE_X_RANGE = (0.50, 0.75)
CUBE_Y_RANGE = (-0.15, 0.15)


def main():
    rng = np.random.default_rng(42)
    successes = 0
    for i in range(N_EPISODES):
        cx = rng.uniform(*CUBE_X_RANGE)
        cy = rng.uniform(*CUBE_Y_RANGE)
        ep = record_pick_episode(ROOT, episode_idx=i, cube_xy=(cx, cy))
        if ep["success"]:
            write_episode(ROOT, ep)
            successes += 1
            print(f"  episode {i}: SUCCESS (cube at ({cx:.2f},{cy:.2f}))")
        else:
            print(f"  episode {i}: FAIL (cube at ({cx:.2f},{cy:.2f}))")

    print(f"\nDone: {successes}/{N_EPISODES} successful episodes recorded to {ROOT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run recording script in background**

Run: `cd /home/aditya/bude_vla && unset PYTHONPATH && PYTHONPATH=/home/aditya/bude_vla/src nohup /home/aditya/.bude-venv/bin/python scripts/record_pick_episodes.py > data/pick_recording.log 2>&1 &`
Expected: writes episodes to `data/pick_v3/`

- [ ] **Step 3: Commit**

```bash
cd /home/aditya/bude_vla && git add scripts/record_pick_episodes.py && git commit -m "feat(scripts): batch pick-and-place recording with randomized cube positions"
```

---

### Task 5: Train on pick data

**Files:**
- Modify: `scripts/train.py` (add `--task pick` shortcut)

- [ ] **Step 1: Add pick task to train.py**

Add `data/pick_v3` path when `--task pick` is passed:

```python
TASK_ROOTS = {
    "reach": ["/home/aditya/bude_vla/data/reach_v3"],
    "push": ["/home/aditya/bude_vla/data/push_v3"],
    "pick": ["/home/aditya/bude_vla/data/pick_v3"],
}
```

Add argparse entry and plumb into `train()`.

- [ ] **Step 2: Run training (5k steps)**

Run: `cd /home/aditya/bude_vla && unset PYTHONPATH && PYTHONPATH=/home/aditya/bude_vla/src nohup /home/aditya/.bude-venv/bin/python scripts/train.py --task pick --n-steps 5000 --batch-size 32 --save-every 1000 > data/pick_training.log 2>&1 &`
Expected: loss drops over steps, checkpoints saved every 1000.

- [ ] **Step 3: Commit**

```bash
cd /home/aditya/bude_vla && git add scripts/train.py && git commit -m "feat(train): add --task pick shortcut for pick-and-place data"
```

---

### Task 6: Render eval rollout MP4

**Files:**
- Create: `scripts/render_pick_rollout.py`

- [ ] **Step 1: Write rollout renderer script**

Load checkpoint, create MuJoCo env, run policy inference step-by-step, render frames, pipe to MP4.

- [ ] **Step 2: Run render**

Run: `cd /home/aditya/bude_vla && unset PYTHONPATH && PYTHONPATH=/home/aditya/bude_vla/src MUJOCO_GL=glfw XDG_RUNTIME_DIR=/tmp DISPLAY=:1 /home/aditya/.bude-venv/bin/python scripts/render_pick_rollout.py`
Expected: writes `demos/videos/pick_rollout.mp4`

- [ ] **Step 3: Commit**

```bash
cd /home/aditya/bude_vla && git add scripts/render_pick_rollout.py && git commit -m "feat(scripts): eval rollout renderer for trained pick-and-place policy"
```
