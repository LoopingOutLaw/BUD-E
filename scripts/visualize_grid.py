"""Render a 2x2 multi-angle grid by injecting 4 distinct cameras into copies of the model.

Camera math: for a "look at" camera at (look_at_target + offset_pos), the x_axis
must lie in the horizontal plane (no z component) and point right-ish. The y_axis
points up. We compute them per spec and write a fresh camera tag for each tile.
"""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import imageio  # noqa: F401
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bude_vla.data.demo_recorder import scripted_reach_step, scripted_push_step


# 4 viewpoints over the working zone (target zone at 0.85, 0)
VIEWS = [
    # (label, pos=(x,y,z), look_at=(lx,ly,lz))
    ("over_shoulder", (0.6, -0.7, 0.7), (0.7, 0.0, 0.45)),  # behind arm, looking forward
    ("side",          (1.0, -0.7, 0.55), (0.6, 0.3, 0.42)),  # looking sideways from right
    ("front",         (1.3, 0.0, 0.55), (0.6, 0.0, 0.45)),  # looking sideways from front-right
    ("top",           (0.7, 0.0, 1.45), (0.7, 0.0, 0.40)),  # top down
]


def _compute_xyaxes(pos, lookat):
    """Given a camera position and target it should look at, return xyaxes.

    x_axis = unit vector pointing RIGHT (perpendicular to forward, horizontal)
    y_axis = up-ish (close to world (0, 0, 1), but adjusted so cross(x,y) points toward target)
    """
    forward = np.asarray(lookat) - np.asarray(pos)
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    # x = right = up cross forward
    x = np.cross(world_up, forward)
    x /= max(np.linalg.norm(x), 1e-6)
    # if forward is parallel to world_up (top-down case), pick a different right vector
    if np.linalg.norm(x) < 0.01:
        x = np.array([0.0, 1.0, 0.0])
    y = np.cross(forward, x)
    return x, y


def _inject_camera(xml_text: str, cam_name: str, pos, xyaxes_t, mode="fixed",
                    resolution="320 240", fovy="60"):
    """Replace the existing 'over_shoulder' camera tag with a new one."""
    pos_str = " ".join(f"{v:.3f}" for v in pos)
    xy_str = " ".join(f"{v:.3f}" for v in xyaxes_t)
    new_tag = (
        f'<camera name="{cam_name}" pos="{pos_str}" '
        f'xyaxes="{xy_str}" mode="{mode}" resolution="{resolution}" fovy="{fovy}"/>'
    )
    # Pattern that matches the existing camera tag with newlines
    import re
    pattern = re.compile(r'<camera name="[^"]*"\s*pos="[^"]*"\s*xyaxes="[^"]*"\s*'
                          r'mode="[^"]*"\s*resolution="[^"]*"\s*fovy="[^"]*"/>',
                          re.MULTILINE)
    if pattern.search(xml_text):
        return pattern.sub(new_tag, xml_text, count=1)
    # Fallback: insert before </worldbody>
    return xml_text.replace("</worldbody>", f"    {new_tag}\n  </worldbody>")


def _roll_to_target(model_xml: str, target_xyz, n_steps=60, viewport=(320, 240),
                    camera="over_shoulder", task="reach"):
    """Roll scripted policy and return last frame."""
    model = mujoco.MjModel.from_xml_string(model_xml)
    d = mujoco.MjData(model)
    qpos = np.zeros(model.nq, dtype=np.float64)
    qpos[0:3] = [0.6, 0.0, 0.435]
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    qpos[7:13] = [0.0, -0.5, 0.9, 0.0, 0.0, 0.0]
    qpos[13] = -0.04
    d.qpos[:] = qpos
    mujoco.mj_forward(model, d)

    renderer = mujoco.Renderer(model, height=viewport[1], width=viewport[0])
    target = np.array(target_xyz, dtype=np.float32)

    # For push need cube_body
    cube_body = None
    target_zone_geom = None
    if task == "push":
        cube_body = next(i for i in range(model.nbody)
                            if model.body(i).name == "cube")
        target_zone_geom = next(i for i in range(model.ngeom)
                                   if model.geom(i).name == "target_zone_disc")

    phase = 0
    for t in range(n_steps):
        ee = np.asarray(d.site_xpos[0], dtype=np.float32)
        if task == "reach":
            action = scripted_reach_step(ee, target, d.qpos,
                                          None, None, nu=model.nu)
        else:
            cube_xyz = np.asarray(d.xpos[cube_body], dtype=np.float32)
            target_pos = np.asarray(model.geom_pos[target_zone_geom], dtype=np.float32)
            action, phase = scripted_push_step(ee, cube_xyz, target_pos, phase,
                                                  nu=model.nu)
            action[-1] = -0.6
        d.ctrl[:] = action
        mujoco.mj_step(model, d)
    renderer.update_scene(d, camera=camera)
    frame = renderer.render()
    renderer.close()
    return frame


def make_grid(out_path="demos/reach_multi_view.png",
               model_path="urdf/ur5e_scene.xml",
               viewport=(180, 135)) -> str:
    """Render the arm at end-of-push from 4 viewpoints in a 2x2 grid."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_xml = Path(model_path).read_text()

    sections = []
    for cam_name, pos, lookat in VIEWS:
        x, y = _compute_xyaxes(pos, lookat)
        xml_with_cam = _inject_camera(base_xml, "over_shoulder", pos, (*x, *y))
        try:
            frame = _roll_to_target(
                xml_with_cam,
                target_xyz=(0.85, 0.0, 0.0),  # cube target
                n_steps=80,  # let push complete
                viewport=viewport,
                task="push",
            )
        except Exception as e:
            print(f"view {cam_name} failed:", e)
            frame = np.zeros((viewport[1], viewport[0], 3), dtype=np.uint8)
        img = Image.fromarray(frame.copy())
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default(size=12)
        except Exception:
            font = ImageFont.load_default()
        draw.rectangle([(2, 2), (104, 22)], fill=(0, 0, 0, 200))
        draw.text((6, 6), cam_name, fill=(255, 240, 0), font=font)
        sections.append(img)

    w, h = sections[0].size
    grid = Image.new("RGB", (w * 2 + 6, h * 2 + 6), (20, 20, 22))
    grid.paste(sections[0], (0, 0))
    grid.paste(sections[1], (w + 6, 0))
    grid.paste(sections[2], (0, h + 6))
    grid.paste(sections[3], (w + 6, h + 6))
    grid.save(str(out_path))
    return str(out_path)


if __name__ == "__main__":
    p = make_grid()
    print("WROTE", p)
