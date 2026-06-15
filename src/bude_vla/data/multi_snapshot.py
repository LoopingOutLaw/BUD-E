"""Multi-arm snapshot grid renderer.

Renders a batch of vmapped MJX states into a single square PIL image so the
training trajectory of N parallel arms can be eyeballed without a live viewer.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import jax
from PIL import Image

from bude_vla.envs.so101_mjx import UR5eMJMJX


def _get_renderer(env: UR5eMJMJX, cell: int):
    """Lazy-init native mujoco.Renderer; reuse it across cells when dimensions match."""
    import mujoco

    r = getattr(env, "_mjx_grid_renderer", None)
    if r is None or r._height != cell or r._width != cell:
        r = mujoco.Renderer(env.model_mj, height=cell, width=cell)
        env._mjx_grid_renderer = r
    return r


def _render_one(env: UR5eMJMJX, state, cell: int, camera: str | None = None) -> np.ndarray:
    """Render a single vmapped MJX state into an (cell, cell, 3) uint8 array.

    Note: an mjx.Data obtained from `mjx.get_data` does NOT have forward-
    kinematics position fields populated. We must run a single native
    `mujoco.mj_forward` on the recovered MjData so xpos/xmat/geom_xpos exist
    before the renderer reads them. Otherwise MJX yields washed-out images.
    """
    import mujoco
    from mujoco import mjx

    renderer = _get_renderer(env, cell)
    d_mj = mjx.get_data(env.model_mj, state)
    mujoco.mj_forward(env.model_mj, d_mj)
    if camera:
        renderer.update_scene(d_mj, camera=camera)
    else:
        renderer.update_scene(d_mj)
    return renderer.render()


def _pick_tile(n_envs: int, tile_req: int) -> int:
    """Return the largest tile <= tile_req with tile^2 <= n_envs (>= 1)."""
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")
    t = int(round(np.sqrt(n_envs)))
    while t * t > n_envs and t > 1:
        t -= 1
    return min(tile_req, t) if tile_req and tile_req * tile_req <= n_envs else t


def snapshot_grid(env: UR5eMJMJX, states, tile: int | None = None, cell: int = 96, camera: str | None = None) -> Image.Image:
    """Render a square grid of arms from a vmapped MJX state batch.

    Args:
        env: UR5eMJMJX instance.
        states: vmapped mjx.Data with leading batch dim.
        tile: optional requested tile size; auto-fit to sqrt(n_envs) if too large.
        cell: pixel size of each rendered cell.
        camera: optional MuJoCo camera name to use for each cell.

    Returns:
        PIL.Image RGB, square. If n_envs doesn't form a perfect square, last few
        cells are blank.
    """
    n_envs = int(states.qpos.shape[0])
    if tile is None:
        t = int(round(np.sqrt(n_envs)))
        while t * t > n_envs and t > 1:
            t -= 1
    else:
        t = _pick_tile(n_envs, tile)

    pad_px = 1
    cells_per_row = t
    grid_w = t * cell + (t + 1) * pad_px
    cells = np.full((t, t, cell, cell, 3), 0, dtype=np.uint8)
    n_to_render = min(n_envs, t * t)
    for i in range(n_to_render):
        single = jax.tree.map(lambda x: x[i], states)
        img = _render_one(env, single, cell, camera=camera)
        r, c = divmod(i, t)
        cells[r, c] = img

    grid = np.full((grid_w, grid_w, 3), 0, dtype=np.uint8)
    for r in range(t):
        for c in range(t):
            y = r * (cell + pad_px) + pad_px
            x = c * (cell + pad_px) + pad_px
            grid[y : y + cell, x : x + cell] = cells[r, c]
    return Image.fromarray(grid)


def write_video(frames: list[Image.Image], path: str, fps: int = 30) -> None:
    """Write a list of PIL images to an MP4 file using ffmpeg (libx264).

    Pipes raw RGB frames to ffmpeg via stdin, one frame at a time.

    Args:
        frames: list of PIL.Image RGB, all same size.
        path: output .mp4 file path.
        fps: frames per second.
    """
    import subprocess

    if not frames:
        raise ValueError("frames must be non-empty")

    w, h = frames[0].size
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "medium",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
