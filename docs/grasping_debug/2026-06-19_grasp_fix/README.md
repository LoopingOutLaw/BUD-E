# Grasp fix — session log, 2026-06-19

This directory contains the **raw verification logs** that justify the fix
landed in commit `9bf50b2`. Everything below was produced by running the
existing diagnostic / verification scripts in this branch with the **same
PYTHONPATH + venv** anywhere, no special environment.

## What was wrong

`scripted_pick_and_place.py` would attach the ball via `GraspController`
when the jaw's qpos was nominally clamped shut — but with the SO-101's
single-asymmetric-jaw geometry, the actuator stalls at `jaw_qpos ≈ 0.475`
whenever it's actively pressing on the 12.5 mm ball (gear=20, no torque
margin against ball+finger friction). The controller was waiting for
`jaw_qpos ≤ 0.30` — an unreachable target — so attach never fired from a
physical gap, and whoever wrote the integration tests was relying on the
recorded trajectory's cached offset (the "floating gap" symptom).

Two ingredients conflated under one constant (`JAW_CLOSED_QPOS_THRESHOLD = 0.30`):

  1. **IK seed** — passed into `_ik_core` so the arm is solved for a
     near-closed-jaw pose. Needs to be the closed jaw value (`0.30`).
  2. **Attach detector** — once the arm is settled, gate on when the
     *real simulated* qpos enters the enclosing window
     (`1.4 → 0.5`), because that's when the ball is geometrically
     enclosed and the geometric gap is small.

When the constant was tuned, the IK seed moved with it; at the wrong
seed value the arm pose was solved for an open jaw and closed jaw
geometry pointed inward, so during the GRASP ramp the moving jaw swept
through the ball instead of around it.

## The fix (commit 9bf50b2)

In `src/bude_vla/grasp.py`:

```python
ATTACH_GAP_TOLERANCE  = 0.025 -> 0.005     # 25 mm -> 5 mm near-contact window
ATTACH_DEBOUNCE_STEPS = 5     -> 3        # attach before the jaw pushes the ball away
JAW_CLOSED_QPOS_THRESHOLD = 0.30 -> 1.40  # gate on enclosing window, not unreachable close
IK_SEED_JAW_QPOS = 0.30                    # NEW: kept separate constant for IK seeding
```

In `src/bude_vla/scripted_pick_and_place.py`:

  - GRASP phase now targets the ball **center** (`GROUND_Z`) instead of
    the static `GROUND_Z + BALL_RADIUS` (north pole).
  - GRASP phase now seeds IK with `IK_SEED_JAW_QPOS` (closed-shape
    geometry), not the raised `JAW_CLOSED_QPOS_THRESHOLD`.
  - GRASP phase **closes the jaw concurrently** with the arm-over-ball
    ramp instead of waiting until after the ramp completes, so the
    qpos is already in the enclosing window when the jaw tip reaches
    the ball instead of arriving there wide open and pushing the ball
    80 mm away while ctrl finally hands it a closing signal.

## What this directory contains

| File                            | Script                         | What it proves                                              |
|---------------------------------|--------------------------------|-------------------------------------------------------------|
| `calibrate_v2.log`              | `scripts/calibrate_grasp_v2.py`| Static (frozen-arm) calibration: at the IK-seeded pose, jaw closes smoothly and attaches at step 26 with `jaw_qpos=1.383`, `gap=1.79 mm`. This is the **physical** attachment event. |
| `diag_attach_gates.log`         | `scripts/diag_attach_gates.py` | Three real scripted episodes with phase / jaw_qpos / gap / contact / streak dumped every 5 steps. Shows the *enclosing window* is reached and attach fires consistently. (242 lines — full trajectory.) |
| `verify_20ep.log`               | `scripts/verify_grasp_fix.py --grasp --bowls --diagnose --episodes 20` | Final pass/fail summary on 20 random cube positions. **15 / 20 attached**, worst carry-gap 3.76 mm, both bowls contain a dropped ball. |

## How to reproduce exactly

```bash
unset PYTHONPATH
cd /home/aditya/bude_vla
MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/calibrate_grasp_v2.py
MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/diag_attach_gates.py
MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/verify_grasp_fix.py --grasp --bowls --diagnose --episodes 20
```

## What is *not* fixed here

The remaining `5 / 20` episodes that attach but don't reach `success=True`
end with `final_dist_to_target ≈ 0.40–0.51 m`. The ball is dropped during
LIFT/MOVE because the kinematic carry drift breaks the
`RELEASE_DRIFT_TOLERANCE = 0.012` gate too early or the gripper release
fires off the wrong phase transition. This is a separate downstream bug
and was deliberately left outside the scope of this fix; it can be
addressed by tightening `RELEASE_DRIFT_TOLERANCE` or replacing the
kinematic carry with friction-aware mjx contact gating.

The "floating gap" / "magic 10 cm pick-up" user-visible bug is fixed.
