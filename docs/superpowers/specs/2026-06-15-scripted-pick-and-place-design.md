# BUD-E Stage 1: Scripted Pick-and-Place (one-cube, one-target)

## Goal

Build a scripted policy that drives BUD-E's UR5e-style arm through a full
pick-and-place demo — reach, descend, grip, lift, translate, release — for a
single red cube placed at a randomized (x, y) on the table, target pose
fixed at (0.85, 0, ~0.421). Record ~100 clean episodes, train BUD-E, render
one polished eval-rollout MP4 — proves end-to-end: data → train → inference.

End-vision (later stages, **out of scope here**): user puts any object at any
pose; BUD-E picks and drops it where asked. Stage 1 ships the spine.

## Scene

Existing XML: `urdf/ur5e_scene.xml`. Cube is a 5cm red box on a
freejoint, currently at qpos[0:7] starting at (0.6, 0, 0.445). The arm has
6 hinge joints (qpos[7:13]) and 1 slider `finger_left` (qpos[13]) coupled
via `equality` joint to `finger_right` (qpos[14]). 7 actuators total.

`bude_vla.data.lerobot_v3.META` already defines `observation.state` as 8-dim:
arm qpos[7:13] (6 floats) + gripper finger_left qpos (1 float) + finger_right
qpos (1 float). Use that.

## Pick-and-place state machine

Single phase variable `phase ∈ {0..5}`:

| Phase | Name     | Action semantics                                                            | Done condition                         |
|-------|----------|-----------------------------------------------------------------------------|----------------------------------------|
| 0     | APPROACH | Drive EE to (cube.x - 0.07, cube.y, cube.z + 0.20): hover above-behind     | within 3 cm AND planarity acceptable   |
| 1     | DESCEND  | Drive EE to (cube.x - 0.05, cube.y, cube.z + 0.10): closer, slightly tilted| within 2 cm AND EE z > cube.z + 0.05   |
| 2     | GRIP     | Apply ctrl=+1 to gripper_motor until both fingers touch cube or 30 steps   | either finger contact or step budget   |
| 3     | LIFT     | Drive EE to (cube.x, cube.y, cube.z + 0.25) while physically attached      | within 5 cm                            |
| 4     | MOVE     | Drive EE to (target.x, target.y, target.z + 0.10)                          | within 5 cm                            |
| 5     | RELEASE  | Apply ctrl=-1 to gripper_motor; let cube fall onto target                  | either finger re-open or 30 steps      |

Episode length budget: 220 steps. After that, force phase=END → terminate.

## Architecture (3 new modules)

### 1. `bude_vla/ik.py` — small inverse-kinematics solver

CPython, not JAX. Uses MuJoCo's `mj_kinematics` to compute the EE site pose
from current qpos, then runs **Jacobian-transpose IK** in pure numpy:

```
J = mj_jacSite(model, data, ee_site)
err_world = target_xyz - ee_xyz
err_local = R.T @ err_world          # rotate error into site frame
dq = alpha * (J.T @ err_local)
qpos[7:13] += dq[6:12]                # only the 6 arm joints
qpos[7:13] = clip(qpos[7:13], -pi, pi)
```

Why Jacobian-transpose over nullspace/CG: stable, ~5 LOC, good enough for
scripted demos at scripted speeds. We are not chasing IK perfection — we
want the arm to *look* like it knows where the cube is.

Public API:

```python
def solve_ik_to_xyz(model, data, target_xyz, current_qpos,
                    site_name="ee_center", max_step=0.05,
                    pos_tol=0.005, max_iters=20) -> np.ndarray:
    """Return new qpos[7:13] that moves EE toward target_xyz.
    Does not call mj_step — caller decides when to integrate."""
```

### 2. `bude_vla/scripted_pick_and_place.py` — the policy state machine

```python
class ScriptedPickAndPlace:
    def __init__(self, model, data, cube_start_xy, target_xy=(0.85, 0.0)):
        self.phase = 0
        self.cube_start_xy = cube_start_xy
        self.target_xy = target_xy
        ...

    def step(self, model, data) -> tuple[np.ndarray, bool, dict]:
        """Returns (ctrl[7], done, info). One MuJoCo substep."""
```

Internally calls `solve_ik_to_xyz` for approach/descend/lift/move, and
forces the gripper actuator in grip/release phases. Counts steps per phase,
auto-advances on done condition.

### 3. `bude_vla/data/pick_recorder.py` — episode collector

Mirrors `cpu_recorder.py`'s structure but replaces the random-reach PD with
`ScriptedPickAndPlace`. Reads `INSTRUCTION = "pick up the red cube and
place it in the blue target zone"`. Records:

- `images`: (T, 64, 64, 3) uint8 from `front_top` camera
- `proprio`: (T, 8) float32 — qpos[7:15]
- `actions`: (T, 7) float32 — the ctrl actually applied
- `instruction`: "pick up the red cube and place it in the blue target zone"
- `cube_xy_history`: (T, 2) for diagnostics (success rate)

Target: 100 successful episodes, each ~150 steps. ~15 k frames total.

**Success criterion**: cube ends episode within 8 cm of target (x, y),
z still on table (> 0.4 m). Failures → recorded as failure categorically
(success=False) and skipped from training data.

## Render

`bude_vla.render.pick_rollout_to_mp4()` — single-file, ~120 LOC. Runs the
trained policy in MJX for one rollout, breaks `cfg.chunk_size` actions into
individual MuJoCo steps, renders `front_top` cam every step, pipe-encodes
to MP4 at 30 fps on `MUJOCO_GL=glfw`/Intel iGPU (:

## Risks and how we mitigate

- **Grasp slips**: the cube has freejoint, fingers are 1 cm wide, 5 cm cube.
  Grasp is fragile. Mitigation: phase 2 holds fingers just closed against
  cube for the full 30-step budget, not "first contact". If it slips, we
  record the failure but **don't** ship success=0 frames to training.
- **PD-overshoot at high speeds**: 0.05 m Jakobian step is plenty slow;
  IK runs max 20 substeps per env-step, target tolerance 5 mm.
- **Cube "blipping" through table**: cube + table are both `box` with
  `condim=4, solref=0.02 1.0` (configured in urdf). Already verified not to
  sink in the existing reach/push demos.
- **MJX rollout render**: stick to MJX for batched eval (consistent with
  training) and only use `mujoco.MjModel` for the IK-driven scripted
  recording (matches `cpu_recorder.py`).
- **JAX on dGPU while MJX eval renders**: training is ~0.14 GB; rendering on
  Intel iGPU (:1, `MUJOCO_GL=glfw`) leaves dGPU untouched.

## Success criteria (what "Stage 1 done" means)

- `test_scripted_policy_reaches_cube_at_known_pose` passes
- `test_scripted_policy_picks_and_releases_cube_on_target` passes
- 100 successful pick-and-place episodes on disk under `data/pick_v3/`
- `BUDETrainingDataset(data/pick_v3)` loads end-to-end
- `train.py --task pick --n-steps 5000` loss goes from ~0.3 to < 0.1
- One MP4 at `demos/videos/pick_rollout.mp4` showing the trained policy
  doing the task on a *novel* cube position (not seen at train time)

## Explicitly out of scope (Stage 2+)

- Multiple objects in scene
- Varying object shapes/colors/sizes
- Free-form target position (currently fixed)
- Text-conditioned target drop (currently fixed instruction)
- Multi-DOF cube orientation (currently cube stays axis-aligned)
- Real-time user input → inference loop (we record demos, we don't run teleop)
- Hardware deploy (SO-101)
- Wild scene generalization with random distractor objects

Each of those is a separate spec/plan.
