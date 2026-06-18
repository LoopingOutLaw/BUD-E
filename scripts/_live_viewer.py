"""Live viewer for MuJoCo cameras.

Step the arm with sliders (or scripted jiggle) and see what each camera renders.
Saves tiled PNG and writes per-camera PNGs to /tmp/live_cam_*.png each call.

Run:
    unset PYTHONPATH && MUJOCO_GL=glfw /home/aditya/.bude-venv/bin/python scripts/_live_viewer.py
"""
from __future__ import annotations
import mujoco
import numpy as np
import imageio.v3 as iio
import os
import sys
import time

XML = '/home/aditya/bude_vla/urdf/ur5e_scene.xml'
JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow',
               'forearm_roll', 'wrist_pitch', 'wrist_roll']


def reset(model, data, lift=-0.5, extend=0.6):
    """Place arm in a descend pose near the cube."""
    mujoco.mj_resetData(model, data)
    jids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in JOINT_NAMES}
    d_qpos = {
        'shoulder_lift': lift,
        'elbow': extend,
    }
    for jname, jid in jids.items():
        if jname in d_qpos:
            data.qpos[model.jnt_qposadr[jid]] = d_qpos[jname]
    mujoco.mj_forward(model, data)
    return data


def render_all(model, data, out_dir='/tmp'):
    """Render every camera, save PNGs, return a 2x3 tile."""
    paths = []
    images = []
    H = W = 224
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        r = mujoco.Renderer(model, H, W)
        r.update_scene(data, camera=i)
        img = r.render()
        path = f'{out_dir}/live_cam_{name}.png'
        iio.imwrite(path, img)
        paths.append(path)
        images.append(img)
    # Build 3x2 tile
    rows = []
    for k in range(0, len(images), 3):
        row_imgs = images[k:k+3]
        # pad if needed
        while len(row_imgs) < 3:
            row_imgs.append(np.zeros((H, W, 3), dtype=np.uint8))
        rows.append(np.concatenate(row_imgs, axis=1))
    tile = np.concatenate(rows, axis=0)
    tile_path = f'{out_dir}/live_cam_tile.png'
    iio.imwrite(tile_path, tile)
    print(f'  -> {tile_path}')
    for p in paths:
        print(f'     {p}')
    return tile, paths


def main():
    print(f'Loading {XML}')
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    print(f'Cameras: {m.ncam}')
    for i in range(m.ncam):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i)
        pos = m.cam_pos[i]
        fovy = m.cam_fovy[i]
        mode = m.cam_mode[i]
        print(f'  [{i}] {name:18s} pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}) '
              f'fovy={fovy:.0f}° mode={mode}')

    # Interactive: read arm states from stdin or use scripted jiggle
    if len(sys.argv) > 1 and sys.argv[1] == 'sweep':
        # Scripted sweep: vary lift/extend across many poses
        for lift in np.linspace(-1.4, -0.3, 6):
            for extend in np.linspace(0.4, 1.6, 6):
                reset(m, d, lift=lift, extend=extend)
                print(f'\n=== lift={lift:+.2f}  extend={extend:+.2f} ===')
                render_all(m, d)
        return

    # Default: single render after reset
    reset(m, d)
    render_all(m, d)


if __name__ == '__main__':
    main()
