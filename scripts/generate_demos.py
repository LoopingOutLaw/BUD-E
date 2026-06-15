"""Comprehensive visualizer: render arm demos from multiple perspectives."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bude_vla.envs.so101_mjx import UR5eMJMJX
from bude_vla.data.demo_recorder import scripted_reach_step
from bude_vla.data.scripted_policies import scripted_push_step


def add_overlay(img_array, text, font_size=12):
    """Add text overlay to a numpy image array."""
    img = Image.fromarray(img_array.copy())
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    
    # Semi-transparent black bar at top
    draw.rectangle([(0, 0), (img.width, 24)], fill=(0, 0, 0, 200))
    draw.text((6, 4), text, fill=(255, 255, 0), font=font)
    return np.asarray(img)


def render_static_poses(out_dir="demos/poses"):
    """Render static poses from the default camera with different configurations."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    env = UR5eMJMJX()
    d = mujoco.MjData(env.model_mj)
    
    # Pose configurations
    poses = {
        "home": [0.0, -0.5, 0.9, 0.0, 0.0, 0.0],  # default
        "extended": [0.3, -1.0, 1.5, 0.0, 0.0, 0.0],  # arm extended forward
        "raised": [0.0, -1.5, 0.5, 0.0, 0.0, 0.0],  # arm raised high
        "reach_right": [-0.5, -0.5, 0.9, 0.0, 0.0, 0.0],  # reaching right
        "grasp_ready": [0.0, -0.3, 1.2, 0.0, 0.0, 0.0],  # ready to grasp
    }
    
    renderer = mujoco.Renderer(env.model_mj, height=480, width=640)
    outputs = []
    
    for name, joint_pos in poses.items():
        # Setup initial pose
        d.qpos[:] = 0.0
        d.qpos[0:3] = [0.6, 0.0, 0.435]  # cube
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[7:13] = joint_pos
        d.qpos[13] = -0.04  # gripper open
        
        mujoco.mj_forward(env.model_mj, d)
        
        renderer.update_scene(d)
        rgb = renderer.render()
        
        out_path = out_dir / f"arm_{name}.png"
        Image.fromarray(rgb).save(str(out_path))
        outputs.append(str(out_path))
        print(f"Saved: {out_path}")
    
    renderer.close()
    return outputs


def run_reach_demo(n_steps=80, out_path="demos/reach_demo.gif", vision_w=320, vision_h=240):
    """Scripted reach demo with trajectory visualization."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    env = UR5eMJMJX()
    target = np.array([0.70, 0.0, 0.55], dtype=np.float32)
    
    # Setup initial pose
    qpos = np.zeros(env.model_mj.nq, dtype=np.float64)
    qpos[0:3] = [0.6, 0.0, 0.435]
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    qpos[7:13] = [0.0, -0.5, 0.9, 0.0, 0.0, 0.0]
    qpos[13] = -0.02
    
    d = mujoco.MjData(env.model_mj)
    d.qpos[:] = qpos
    mujoco.mj_forward(env.model_mj, d)
    
    renderer = mujoco.Renderer(env.model_mj, height=vision_h, width=vision_w)
    frames = []
    
    ctrl_lo, ctrl_hi = env.action_bounds()
    
    for t in range(n_steps):
        ee = np.asarray(d.site_xpos[0], dtype=np.float32)
        distance = float(np.linalg.norm(ee - target))
        
        action = scripted_reach_step(ee, target, d.qpos, ctrl_lo, ctrl_hi, nu=env.model_mj.nu)
        d.ctrl[:] = action
        mujoco.mj_step(env.model_mj, d)
        
        renderer.update_scene(d)
        rgb = renderer.render()
        
        # Add overlay
        label = f"Step {t:2d} | Distance: {distance:.3f}m | Target: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})"
        rgb = add_overlay(rgb, label, font_size=10)
        frames.append(rgb)
        
        if distance < 0.04:
            # Hold position for a few frames
            for _ in range(15):
                mujoco.mj_step(env.model_mj, d)
                renderer.update_scene(d)
                rgb = renderer.render()
                label = f"Step {t+1:2d} | REACH SUCCESS! | Distance: {distance:.3f}m"
                rgb = add_overlay(rgb, label, font_size=10)
                frames.append(rgb)
            break
    
    renderer.close()
    imageio.mimsave(str(out_path), frames, fps=15)
    print(f"Saved: {out_path}")
    return str(out_path)


def run_push_demo(n_steps=120, out_path="demos/push_demo.gif", vision_w=320, vision_h=240):
    """Scripted push demo."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    env = UR5eMJMJX()
    cube_start_y = -0.05
    
    # Get target zone position
    target_zone_geom = next(i for i in range(env.model_mj.ngeom)
                              if env.model_mj.geom(i).name == "target_zone_disc")
    target_pos = np.asarray(env.model_mj.geom_pos[target_zone_geom], dtype=np.float32)
    
    cube_body = next(i for i in range(env.model_mj.nbody)
                       if env.model_mj.body(i).name == "cube")
    
    # Setup
    qpos = np.zeros(env.model_mj.nq, dtype=np.float64)
    qpos[0:3] = [0.6, cube_start_y, 0.435]
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    qpos[7:13] = [0.0, -0.5, 0.9, 0.0, 0.0, 0.0]
    qpos[13] = -0.04
    
    d = mujoco.MjData(env.model_mj)
    d.qpos[:] = qpos
    mujoco.mj_forward(env.model_mj, d)
    
    renderer = mujoco.Renderer(env.model_mj, height=vision_h, width=vision_w)
    frames = []
    phase = 0
    
    for t in range(n_steps):
        ee = np.asarray(d.site_xpos[0], dtype=np.float32)
        cube_xyz = np.asarray(d.xpos[cube_body], dtype=np.float32)
        distance = float(np.linalg.norm(cube_xyz[:2] - target_pos[:2]))
        
        action, phase = scripted_push_step(ee, cube_xyz, target_pos, phase, nu=env.model_mj.nu)
        action[-1] = -0.6  # Open gripper
        d.ctrl[:] = action
        mujoco.mj_step(env.model_mj, d)
        
        renderer.update_scene(d)
        rgb = renderer.render()
        
        label = f"Step {t:2d} | Phase: {phase} | Cube: ({cube_xyz[0]:.2f}, {cube_xyz[1]:.2f}) | Dist: {distance:.3f}m"
        rgb = add_overlay(rgb, label, font_size=9)
        frames.append(rgb)
        
        if distance < 0.05:
            # Success hold
            for _ in range(20):
                mujoco.mj_step(env.model_mj, d)
                renderer.update_scene(d)
                rgb = renderer.render()
                label = f"Step {t+1:2d} | PUSH SUCCESS! | Cube at target!"
                rgb = add_overlay(rgb, label, font_size=10)
                frames.append(rgb)
            break
    
    renderer.close()
    imageio.mimsave(str(out_path), frames, fps=15)
    print(f"Saved: {out_path}")
    return str(out_path)


def run_pick_place_demo(n_steps=150, out_path="demos/pick_place_demo.gif", vision_w=320, vision_h=240):
    """Scripted pick and place demo."""
    out_path = Path(out_path)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    env = UR5eMJMJX()
    
    # Target: pick cube at (0.6, -0.05, 0.435) and place at (0.85, 0.0, 0.435)
    pick_pos = np.array([0.6, -0.05, 0.435], dtype=np.float32)
    place_pos = np.array([0.85, 0.0, 0.435], dtype=np.float32)
    
    cube_body = next(i for i in range(env.model_mj.nbody) if env.model_mj.body(i).name == "cube")
    
    # Setup initial pose
    qpos = np.zeros(env.model_mj.nq, dtype=np.float64)
    qpos[0:3] = pick_pos
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    qpos[7:13] = [0.0, -0.5, 0.9, 0.0, 0.0, 0.0]
    qpos[13] = -0.04  # Open
    
    d = mujoco.MjData(env.model_mj)
    d.qpos[:] = qpos
    mujoco.mj_forward(env.model_mj, d)
    
    renderer = mujoco.Renderer(env.model_mj, height=vision_h, width=vision_w)
    frames = []
    
    ctrl_lo, ctrl_hi = env.action_bounds()
    
    phase = "approach"  # approach -> grasp -> lift -> move -> place -> release
    grasp_count = 0
    
    for t in range(n_steps):
        ee = np.asarray(d.site_xpos[0], dtype=np.float32)
        cube_xyz = np.asarray(d.xpos[cube_body], dtype=np.float32)
        
        if phase == "approach":
            # Move to above cube
            target = pick_pos + np.array([0, 0, 0.15], dtype=np.float32)
            action = scripted_reach_step(ee, target, d.qpos, ctrl_lo, ctrl_hi, nu=env.model_mj.nu)
            d.ctrl[:] = action
            if np.linalg.norm(ee - target) < 0.05:
                phase = "descend"
        elif phase == "descend":
            target = pick_pos + np.array([0, 0, 0.03], dtype=np.float32)
            action = scripted_reach_step(ee, target, d.qpos, ctrl_lo, ctrl_hi, nu=env.model_mj.nu)
            d.ctrl[:] = action
            if np.linalg.norm(ee - target) < 0.03:
                phase = "grasp"
        elif phase == "grasp":
            # Close gripper
            action = np.zeros(env.model_mj.nu)
            action[-1] = 0.5  # Close
            d.ctrl[:] = action
            grasp_count += 1
            if grasp_count > 10:
                phase = "lift"
        elif phase == "lift":
            target = pick_pos + np.array([0, 0, 0.2], dtype=np.float32)
            action = scripted_reach_step(ee, target, d.qpos, ctrl_lo, ctrl_hi, nu=env.model_mj.nu)
            action[-1] = 0.5  # Keep grasped
            d.ctrl[:] = action
            if np.linalg.norm(ee - target) < 0.05:
                phase = "move"
        elif phase == "move":
            target = place_pos + np.array([0, 0, 0.2], dtype=np.float32)
            action = scripted_reach_step(ee, target, d.qpos, ctrl_lo, ctrl_hi, nu=env.model_mj.nu)
            action[-1] = 0.5
            d.ctrl[:] = action
            if np.linalg.norm(ee - target) < 0.05:
                phase = "place"
        elif phase == "place":
            target = place_pos + np.array([0, 0, 0.03], dtype=np.float32)
            action = scripted_reach_step(ee, target, d.qpos, ctrl_lo, ctrl_hi, nu=env.model_mj.nu)
            action[-1] = 0.5
            d.ctrl[:] = action
            if np.linalg.norm(ee - target) < 0.03:
                phase = "release"
        elif phase == "release":
            action = np.zeros(env.model_mj.nu)
            action[-1] = -0.6  # Open
            d.ctrl[:] = action
            # Hold for a bit then done
            if t > n_steps - 20:
                phase = "done"
        
        mujoco.mj_step(env.model_mj, d)
        
        renderer.update_scene(d)
        rgb = renderer.render()
        
        label = f"Step {t:2d} | Phase: {phase} | Gripper: {'Closed' if d.ctrl[-1] > 0 else 'Open'}"
        rgb = add_overlay(rgb, label, font_size=10)
        frames.append(rgb)
        
        if phase == "done":
            break
    
    renderer.close()
    imageio.mimsave(str(out_path), frames, fps=15)
    print(f"Saved: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate arm simulation GIFs and PNGs")
    ap.add_argument("--all", action="store_true", help="Generate all demos")
    ap.add_argument("--reach", action="store_true", help="Generate reach demo")
    ap.add_argument("--push", action="store_true", help="Generate push demo")
    ap.add_argument("--pick-place", action="store_true", help="Generate pick and place demo")
    ap.add_argument("--poses", action="store_true", help="Generate static poses from multiple angles")
    args = ap.parse_args()
    
    if args.all or not any([args.reach, args.push, args.pick_place, args.poses]):
        print("Generating all demos...")
        run_reach_demo()
        run_push_demo()
        run_pick_place_demo()
        render_static_poses()
    else:
        if args.reach:
            run_reach_demo()
        if args.push:
            run_push_demo()
        if args.pick_place:
            run_pick_place_demo()
        if args.poses:
            render_static_poses()
