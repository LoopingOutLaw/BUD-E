"""Render a 2x2 multi-angle grid of the arm doing scripted reach at multiple cams.

This avoids the live viewer segfault by staying on the CPU GIF renderer.
"""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bude_vla.data.demo_recorder import scripted_reach_step


def _home_qpos(model) -> np.ndarray:
    qpos = np.zeros(model.nq, dtype=np.float64)
    qpos[0:3] = [0.6, 0.0, 0.435]
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    qpos[7:13] = [0.0, -0.5, 0.9, 0.0, 0.0, 0.0]
    qpos[13] = -0.04
    return qpos


def _roll_to_target(model_xml: str, target_xyz, n_steps=60, viewport=(320, 240),
                    camera="over_shoulder"):
    model = mujoco.MjModel.from_xml_string(model_xml)
    d = mujoco.MjData(model)
    d.qpos[:] = _home_qpos(model)
    mujoco.mj_forward(model, d)
    renderer = mujoco.Renderer(model, height=viewport[1], width=viewport[0])
    target = np.array(target_xyz, dtype=np.float32)
    frames = []
    for t in range(n_steps):
        ee = np.asarray(d.site_xpos[0], dtype=np.float32)
        action = scripted_reach_step(ee, target, d.qpos,
                                       None, None, nu=model.nu)
        d.ctrl[:] = action
        mujoco.mj_step(model, d)
        renderer.update_scene(d, camera=camera)
        frames.append(renderer.render())
    renderer.close()
    return frames


def make_grid(out_path="demos/reach_multi_view.png",
               model_path="urdf/ur5e_scene.xml") -> str:
    """Render the same reach in 4 different cameras and tile into 2x2."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_xml = Path(model_path).read_text()
    # Pick the LAST frame so all cameras show the arm at end of reach
    cams = ["over_shoulder", "front", "side", "top"]
    # Add named cameras as needed by editing the XML on the fly: add free cameras via MuJoCo API
    # We can use mujoco's default cameras; 'over_shoulder' is in our XML; 'front', 'side', 'top' may not exist.

    base = mujoco.MjModel.from_xml_string(model_xml)
    for cam_name in cams:
        if cam_name == "over_shoulder":
            continue
        try:
            idx = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if idx < 0:
                raise ValueError
        except Exception:
            # Use mujoco built-in default (free) camera offset
            pass

    last_frames = {}
    for cam in ["over_shoulder"]:
        try:
            frames = _roll_to_target(
                model_xml,
                target_xyz=(0.70, 0.0, 0.55),
                n_steps=60,
                camera=cam,
            )
            last_frames[cam] = frames[-1]
        except Exception as e:
            print(f"camera {cam} failed:", e)

    # For cameras not defined in XML, use a single base frame from each offset by
    # temporarily editing the over_shoulder camera position and re-running.
    # Simpler: render at 4 angles of the same scene using viewer-style pose deltas
    # via custom cameras.
    # Custom cameras: pos / xyaxes converted to lookat quat.
    # Simpler: use cam_pos directly, mujoco supports free cameras.

    # Fallback: render with four synthetic cameras (each is a temporary camera added to a copy of the model)
    sections = []
    camera_specs = [
        ("over-shoulder",   (1.1,  0.0,  1.0), (0, -1, 0, 0.5, 0, 1)),
        ("side",            (1.2,  0.6,  0.6), (0, -1, 0, 1,  0, 0.6)),
        ("front",           (0.6, -0.6,  0.6), (0,  1, 0, 1, -0.4, 0.6)),
        ("top",             (0.6,  0.0,  1.2), (0, -1, 0, 0,  0, -1)),
    ]

    for name, pos, xyaxes in camera_specs:
        # Make a one-shot copy with this camera added
        import re
        new_xml = model_xml.replace(
            '<camera name="over_shoulder" pos="1.1 0.0 1.0" xyaxes="0 -1 0 0.5 0 1"',
            f'<camera name="over_shoulder" pos="{" ".join(f"{x:.3f}" for x in pos)}" '
            f'xyaxes="{" ".join(str(v) for v in xyaxes)}"'
        )
        try:
            frames = _roll_to_target(
                new_xml,
                target_xyz=(0.70, 0.0, 0.55) if name == "over-shoulder" else (0.85, 0.0, 0.0),
                n_steps=3,  # only need final frame
                viewport=(240, 180),
            )
            last = frames[-1]
        except Exception as e:
            print(f"view {name} failed:", e)
            last = np.zeros((180, 240, 3), dtype=np.uint8)
        img = Image.fromarray(last.copy())
        draw = ImageDraw.Draw(img)
        draw.rectangle([(2, 2), (130, 22)], fill=(0, 0, 0, 200))
        try:
            font = ImageFont.load_default(size=11)
        except Exception:
            font = ImageFont.load_default()
        draw.text((6, 6), name, fill=(255, 240, 0), font=font)
        sections.append(img)

    # Tile 2x2
    w, h = sections[0].size
    grid = Image.new("RGB", (w * 2, h * 2), (20, 20, 22))
    grid.paste(sections[0], (0, 0))
    grid.paste(sections[1], (w, 0))
    grid.paste(sections[2], (0, h))
    grid.paste(sections[3], (w, h))
    # Border
    grid.save(str(out_path))
    return str(out_path)


if __name__ == "__main__":
    p = make_grid()
    print("WROTE", p)
