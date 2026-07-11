"""Benchmark pick-and-place with camera-only cube localization.

The controller uses:
  - front_top RGB for cube XY
  - fixed camera calibration
  - robot joint state through IK
  - finger-pad contacts for grasp confirmation

Simulator cube pose is read only after control decisions for diagnostics and
success metrics. It is never passed to the controller during a rollout.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from bude_vla.envs.so101_mjx import (
    EXPERT_CONTROL_SUBSTEPS,
    PICK_WORKSPACE_X_RANGE,
    PICK_WORKSPACE_Y_RANGE,
    is_grasping_from_contacts,
    is_touching_cube_from_contacts,
    load_arm_model,
)
from bude_vla.scripted_pick_and_place import ScriptedPickAndPlace
from bude_vla.visual_servo import (
    RedCubeWorldTracker,
    calibrate_red_cube_homography,
)
from eval_pick_ball import parse_cube_positions, reset_arm, reset_cube

SUBSTEPS_PER_FRAME = EXPERT_CONTROL_SUBSTEPS
SUCCESS_THRESHOLD = 0.05


def run_one(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tracker: RedCubeWorldTracker,
    cube_xy: tuple[float, float],
    max_steps: int,
    max_grasp_retries: int,
) -> dict:
    mujoco.mj_resetData(model, data)
    reset_arm(model, data)
    reset_cube(data, cube_xy[0], cube_xy[1])
    for _ in range(50):
        mujoco.mj_step(model, data)

    tracker.reset()
    initial_estimate = tracker.update(data, force=True)
    if initial_estimate is None:
        return {
            "success": False,
            "touch_frames": 0,
            "grasp_frames": 0,
            "steps": 0,
            "initial_error_m": float("nan"),
            "tracking_errors": [],
            "reason": "not_visible",
        }

    expert = ScriptedPickAndPlace(
        model,
        data,
        initial_estimate,
        max_grasp_retries=max_grasp_retries,
        cube_position_provider=tracker,
    )
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
    touch_frames = 0
    grasp_frames = 0
    ever_grasped = False
    tracking_errors: list[float] = []
    reason = "timeout"

    for step in range(max_steps):
        ctrl, _arm_q, done, _info = expert.step(model, data)
        data.ctrl[:] = ctrl
        for _ in range(SUBSTEPS_PER_FRAME):
            mujoco.mj_step(model, data)

        touching = is_touching_cube_from_contacts(model, data) > 0.5
        grasping = is_grasping_from_contacts(model, data) > 0.5
        touch_frames += int(touching)
        grasp_frames += int(grasping)
        ever_grasped = ever_grasped or grasping

        estimate = tracker.last_xy
        if estimate is not None and not ever_grasped:
            true_xy = data.xpos[cube_id, :2]
            tracking_errors.append(float(np.linalg.norm(estimate - true_xy)))

        if done:
            reason = "controller_done"
            break
    else:
        step = max_steps - 1

    cube = data.xpos[cube_id].copy()
    target = data.xpos[target_id].copy()
    success = bool(
        ever_grasped
        and np.linalg.norm(cube[:2] - target[:2]) < SUCCESS_THRESHOLD
    )
    return {
        "success": success,
        "touch_frames": touch_frames,
        "grasp_frames": grasp_frames,
        "steps": step + 1,
        "initial_error_m": float(np.linalg.norm(initial_estimate - np.asarray(cube_xy))),
        "tracking_errors": tracking_errors,
        "reason": reason,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-episodes", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=2200)
    ap.add_argument("--max-grasp-retries", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--calibration-grid", type=int, default=5)
    ap.add_argument("--tracker-smoothing", type=float, default=0.25)
    ap.add_argument("--min-success-rate", type=float, default=0.0,
                    help="Exit nonzero when camera-only success is below this rate.")
    ap.add_argument("--cube-positions", default=None)
    ap.add_argument("--cube-x-range",
                    default=",".join(str(v) for v in PICK_WORKSPACE_X_RANGE))
    ap.add_argument("--cube-y-range",
                    default=",".join(str(v) for v in PICK_WORKSPACE_Y_RANGE))
    args = ap.parse_args()

    model = load_arm_model()
    data = mujoco.MjData(model)
    reset_arm(model, data)
    renderer = mujoco.Renderer(
        model, height=args.img_size, width=args.img_size
    )
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "front_top"
    )
    if camera_id < 0:
        raise RuntimeError("front_top camera is missing")

    x_range = tuple(float(v) for v in args.cube_x_range.split(","))
    y_range = tuple(float(v) for v in args.cube_y_range.split(","))
    homography, calibration = calibrate_red_cube_homography(
        model,
        data,
        renderer,
        camera_id,
        x_range=x_range,
        y_range=y_range,
        grid_size=args.calibration_grid,
    )
    print(
        "calibration: "
        f"points={int(calibration['points'])} "
        f"mean={calibration['mean_error_m'] * 1000:.2f}mm "
        f"p95={calibration['p95_error_m'] * 1000:.2f}mm "
        f"max={calibration['max_error_m'] * 1000:.2f}mm",
        flush=True,
    )
    tracker = RedCubeWorldTracker(
        renderer,
        camera_id,
        homography,
        smoothing=args.tracker_smoothing,
    )

    rng = np.random.default_rng(args.seed)
    fixed_positions = parse_cube_positions(args.cube_positions)
    results = []
    try:
        for ep in range(args.num_episodes):
            if fixed_positions:
                cube_xy = fixed_positions[ep % len(fixed_positions)]
            else:
                cube_xy = (
                    float(rng.uniform(*x_range)),
                    float(rng.uniform(*y_range)),
                )
            result = run_one(
                model,
                data,
                tracker,
                cube_xy,
                args.max_steps,
                args.max_grasp_retries,
            )
            results.append(result)
            errors = result["tracking_errors"]
            track_mm = np.median(errors) * 1000 if errors else float("nan")
            print(
                f"ep {ep:03d} cube=({cube_xy[0]:.3f},{cube_xy[1]:.3f}) "
                f"init_err={result['initial_error_m'] * 1000:.1f}mm "
                f"track_med={track_mm:.1f}mm "
                f"touch={result['touch_frames']} grasp={result['grasp_frames']} "
                f"success={result['success']} steps={result['steps']} "
                f"reason={result['reason']}",
                flush=True,
            )
    finally:
        renderer.close()

    n = len(results)
    successes = sum(int(r["success"]) for r in results)
    touched = sum(int(r["touch_frames"] > 0) for r in results)
    grasped = sum(int(r["grasp_frames"] > 0) for r in results)
    initial_errors = np.asarray([
        r["initial_error_m"] for r in results if np.isfinite(r["initial_error_m"])
    ])
    tracking_errors = np.asarray([
        error for r in results for error in r["tracking_errors"]
    ])
    print("\n=== VISUAL-SERVO PICK BENCH ===")
    print(f"episodes: {n}")
    print(f"success episodes: {successes}/{n} ({successes / max(1, n):.3f})")
    print(f"any_contact episodes: {touched}/{n} ({touched / max(1, n):.3f})")
    print(f"strict_grasp episodes: {grasped}/{n} ({grasped / max(1, n):.3f})")
    if initial_errors.size:
        print(f"median initial localization error: {np.median(initial_errors) * 1000:.2f} mm")
        print(f"p95 initial localization error: {np.percentile(initial_errors, 95) * 1000:.2f} mm")
    if tracking_errors.size:
        print(f"median pre-grasp tracking error: {np.median(tracking_errors) * 1000:.2f} mm")
        print(f"p95 pre-grasp tracking error: {np.percentile(tracking_errors, 95) * 1000:.2f} mm")
    success_rate = successes / max(1, n)
    if success_rate < args.min_success_rate:
        raise SystemExit(
            f"camera-only success {success_rate:.3f} is below required {args.min_success_rate:.3f}"
        )


if __name__ == "__main__":
    main()
