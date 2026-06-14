"""Quick visualizer: render the arm doing scripted tasks, save a GIF."""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bude_vla.envs.so101_mjx import UR5eMJMJX
from bude_vla.data.demo_recorder import scripted_reach_step, INSTRUCTION_BY_TASK
from bude_vla.data.scripted_policies import scripted_push_step


def run_reach_demo(n_steps: int = 60,
                    out_path: str = "demos/reach_demo.gif",
                    target_xyz: tuple[float, float, float] = (0.70, 0.0, 0.55),
                    vision_w: int = 320, vision_h: int = 240) -> str:
    """Roll a scripted reach and write a per-step GIF with overlays."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = UR5eMJMJX()
    target = np.array(target_xyz, dtype=np.float32)

    # qpos layout: [cube_free(7), arm_joints(6), gripper(1):2]
    # 7 = cube xyz at addresses 0..2, quat at 3..6
    # 6 arm joints at 7..12, 1 gripper slide at 13
    # finger_right is an equality-mirror of finger_left, no extra qpos entries (length 15 total).
    qpos = np.zeros(env.model_mj.nq, dtype=np.float64)
    qpos[0:3] = [0.6, 0.0, 0.435]               # cube pose
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]              # cube quat (identity)
    qpos[7] = 0.0                                # shoulder_pan
    qpos[8] = -0.5                               # shoulder_lift
    qpos[9] = 0.9                                # elbow
    qpos[10:13] = [0.0, 0.0, 0.0]                 # forearm_roll, wrist_pitch, wrist_roll
    qpos[13] = -0.02                              # finger_left slider near closed

    d = mujoco.MjData(env.model_mj)
    d.qpos[:] = qpos
    mujoco.mj_forward(env.model_mj, d)

    renderer = mujoco.Renderer(env.model_mj, height=vision_h, width=vision_w)

    frames: list[np.ndarray] = []
    distances: list[float] = []
    annotations: list[str] = []

    ctrl_lo, ctrl_hi = env.action_bounds()
    import jax.numpy as jnp
    from mujoco import mjx as mjxlib
    mj_model = env.model_mj
    mjx_model = env.model
    state = mjxlib.put_data(mj_model, d)

    for t in range(n_steps):
        # Use mujoco CPU-rendered scene from current mujoco.MjData
        ee = np.asarray(d.site_xpos[0], dtype=np.float32)
        distance = float(np.linalg.norm(ee - target))
        distances.append(distance)

        action = scripted_reach_step(ee, target, d.qpos,
                                       ctrl_lo, ctrl_hi, nu=mj_model.nu)
        d.ctrl[:] = action
        mujoco.mj_step(mj_model, d)

        renderer.update_scene(d)
        rgb = renderer.render()
        frames.append(rgb)

        annotations.append(f"step {t}  dist {distance:.3f}")

        if distance < 0.04:
            annotations.append("REACHED!")
            break

    for extra in range(8):
        mujoco.mj_step(mj_model, d)
        renderer.update_scene(d)
        frames.append(renderer.render())
        distances.append(distance)
        annotations.append(f"step {len(frames)-1}  dist {distance:.3f}")

    renderer.close()

    # Annotate frames
    annotated = []
    for i, (f, label) in enumerate(zip(frames, annotations)):
        img = Image.fromarray(f.copy())
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default(size=12) if ImageFont.load_default else ImageFont.load_default()
        except Exception:
            font = None
        draw.rectangle([(2, 2), (250, 22)], fill=(0, 0, 0, 200))
        draw.text((6, 6), label, fill=(255, 240, 0), font=font)
        annotated.append(np.asarray(img))

    imageio.mimsave(str(out_path), annotated, fps=15)
    return str(out_path)


def run_push_demo(n_steps: int = 80,
                   out_path: str = "demos/push_demo.gif",
                   cube_start_y: float = -0.05,
                   target_zone_offset: tuple[float, float] = (0.0, 0.0),
                   vision_w: int = 320, vision_h: int = 240) -> str:
    """Roll a scripted push and write a per-step GIF with overlays."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = UR5eMJMJX()
    target_zone_geom = next(i for i in range(env.model_mj.ngeom)
                              if env.model_mj.geom(i).name == "target_zone_disc")
    target_pos = np.asarray(env.model_mj.geom_pos[target_zone_geom], dtype=np.float32)
    target_pos[0] += target_zone_offset[0]
    target_pos[1] += target_zone_offset[1]

    cube_body = next(i for i in range(env.model_mj.nbody)
                       if env.model_mj.body(i).name == "cube")

    # Default qpos: cube at (0.6, cube_start_y, 0.435); arm in semi-extended pose
    qpos = np.zeros(env.model_mj.nq, dtype=np.float64)
    qpos[0:3] = [0.6, cube_start_y, 0.435]
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    qpos[7:13] = [0.0, -0.5, 0.9, 0.0, 0.0, 0.0]
    qpos[13] = -0.04  # gripper fully open

    d = mujoco.MjData(env.model_mj)
    d.qpos[:] = qpos
    mujoco.mj_forward(env.model_mj, d)

    renderer = mujoco.Renderer(env.model_mj, height=vision_h, width=vision_w)

    frames: list[np.ndarray] = []
    annotations: list[str] = []

    phase = 0
    distances: list[float] = []
    success = False

    for t in range(n_steps):
        ee = np.asarray(d.site_xpos[0], dtype=np.float32)
        cube_xyz = np.asarray(d.xpos[cube_body], dtype=np.float32)
        distance = float(np.linalg.norm(cube_xyz[:2] - target_pos[:2]))
        distances.append(distance)

        action, phase = scripted_push_step(ee, cube_xyz, target_pos, phase,
                                              nu=env.model_mj.nu)
        # Open gripper during push (last slider)
        action[-1] = -0.6
        d.ctrl[:] = action
        mujoco.mj_step(env.model_mj, d)

        renderer.update_scene(d)
        frames.append(renderer.render())
        annotations.append(
            f"step {t}  phase {phase}  cube ({cube_xyz[0]:.2f},{cube_xyz[1]:.2f})  dist {distance:.3f}"
        )

        if distance < 0.05:
            annotations[-1] += "  PUSHED!"
            success = True
            # Hold for a bit to show success
            for _ in range(15):
                mujoco.mj_step(env.model_mj, d)
                renderer.update_scene(d)
                frames.append(renderer.render())
                cube_xyz = np.asarray(d.xpos[cube_body], dtype=np.float32)
                annotations.append(
                    f"step {len(frames)-1}  cube ({cube_xyz[0]:.2f},{cube_xyz[1]:.2f})  dist {distance:.3f}  PUSHED!"
                )
            break

    renderer.close()

    annotated = []
    for label, f in zip(annotations, frames):
        img = Image.fromarray(f.copy())
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default(size=12) if hasattr(ImageFont, "load_default") else None
        except Exception:
            font = ImageFont.load_default()
        # Header bar
        draw.rectangle([(2, 2), (310, 22)], fill=(0, 0, 0, 200))
        draw.text((6, 6), label[:64], fill=(255, 240, 0), font=font)
        annotated.append(np.asarray(img))

    imageio.mimsave(str(out_path), annotated, fps=15)
    return str(out_path)


def render_pose(out_path: str = "demos/arm_home.png",
                 vision_w: int = 640, vision_h: int = 480) -> str:
    """Render a single static PNG of the arm in its default home pose."""
    import mujoco
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = UR5eMJMJX()
    d = mujoco.MjData(env.model_mj)
    d.qpos[0:3] = [0.6, 0.0, 0.435]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[7] = 0.0
    d.qpos[8] = -0.5
    d.qpos[9] = 0.9
    mujoco.mj_forward(env.model_mj, d)
    r = mujoco.Renderer(env.model_mj, height=vision_h, width=vision_w)
    r.update_scene(d)
    rgb = r.render()
    r.close()
    Image.fromarray(rgb).save(str(out_path))
    return str(out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["reach", "static", "push"], default="reach")
    ap.add_argument("--out", default="demos/reach_demo.gif")
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()
    if args.mode == "reach":
        p = run_reach_demo(n_steps=args.steps, out_path=args.out)
    elif args.mode == "push":
        p = run_push_demo(n_steps=args.steps, out_path=args.out)
    else:
        p = render_pose(out_path="demos/arm_home.png")
    print("WROTE", p)
