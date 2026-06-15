"""Tests for hi-res grid renderer + MP4 video writing."""
import os
import subprocess
from pathlib import Path

import numpy as np
import jax
from PIL import Image

from bude_vla.envs.so101_mjx import UR5eMJMJX
from bude_vla.data.multi_snapshot import snapshot_grid


def _repeat(env, n: int):
    states = env.reset()
    return jax.tree.map(lambda x: jax.numpy.repeat(x[None, ...], n, axis=0), states)


def test_hi_res_grid_cell_size_respected():
    """Large cell=320 produces a large image with rich pixel content."""
    env = UR5eMJMJX()
    states = _repeat(env, 4)
    cell = 320
    img = snapshot_grid(env, states, tile=2, cell=cell)
    expected = 2 * cell + 3 * 1
    assert img.size == (expected, expected), f"got {img.size}, want {(expected, expected)}"
    arr = np.asarray(img)
    assert arr.std() > 50.0, f"hi-res render should have rich pixel std (got {arr.std():.1f})"


def test_solo_over_shoulder_camera_at_high_res():
    """A single-arm over_shoulder close-up at 640x640."""
    env = UR5eMJMJX()
    states = _repeat(env, 1)
    img = snapshot_grid(env, states, tile=1, cell=640, camera="over_shoulder")
    assert img.size == (642, 642)
    arr = np.asarray(img)
    assert arr.std() > 50.0, f"over_shoulder hero shot should be visually rich (std {arr.std():.1f})"


def test_pov_camera_at_high_res():
    """POV camera close-up at 640x640."""
    env = UR5eMJMJX()
    states = _repeat(env, 1)
    img = snapshot_grid(env, states, tile=1, cell=640, camera="pov")
    assert img.size == (642, 642)
    arr = np.asarray(img)
    assert arr.std() > 50.0, f"pov shot should be visually rich (std {arr.std():.1f})"


def test_wrist_camera_at_high_res():
    """Wrist camera close-up at 640x640."""
    env = UR5eMJMJX()
    states = _repeat(env, 1)
    img = snapshot_grid(env, states, tile=1, cell=640, camera="wrist")
    assert img.size == (642, 642)
    arr = np.asarray(img)
    assert arr.std() > 50.0, f"wrist shot should be visually rich (std {arr.std():.1f})"


def test_write_video_mp4(tmp_path):
    """write_video produces a valid MP4 from a list of PIL frames."""
    from bude_vla.data.multi_snapshot import write_video

    env = UR5eMJMJX()
    states = _repeat(env, 1)
    frames = [snapshot_grid(env, states, tile=1, cell=320, camera="pov") for _ in range(5)]
    out = tmp_path / "test.mp4"
    write_video(frames, str(out), fps=10)
    assert out.exists(), "MP4 file should exist"
    assert out.stat().st_size > 500, "MP4 should not be empty"


def test_write_video_mp4_playback_dims(tmp_path):
    """Written MP4 has correct width/height and frame count via ffprobe."""
    from bude_vla.data.multi_snapshot import write_video

    env = UR5eMJMJX()
    states = _repeat(env, 1)
    frames = [snapshot_grid(env, states, tile=1, cell=160, camera="pov") for _ in range(8)]
    out = tmp_path / "dims.mp4"
    write_video(frames, str(out), fps=15)
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames",
         "-of", "csv=p=0", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"ffprobe failed: {r.stderr}"
    parts = r.stdout.strip().split(",")
    w, h, nframes = int(parts[0]), int(parts[1]), int(parts[2])
    assert w == 162 and h == 162, f"got {w}x{h}"
    assert nframes == 8, f"got {nframes} frames"
