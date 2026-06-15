"""Tests for multi-arm snapshot grid renderer."""
import numpy as np
from PIL import Image
from bude_vla.envs.so101_mjx import UR5eMJMJX
from bude_vla.data.multi_snapshot import snapshot_grid


def test_snapshot_grid_returns_pil_image():
    env = UR5eMJMJX()
    n_envs = 8
    states = env.reset()
    states = __import__("jax").tree.map(
        lambda x: __import__("jax").numpy.repeat(x[None, ...], n_envs, axis=0),
        states,
    )
    img = snapshot_grid(env, states, tile=8)
    assert isinstance(img, Image.Image)


def test_snapshot_grid_square_dimensions():
    env = UR5eMJMJX()
    states = env.reset()
    states = __import__("jax").tree.map(
        lambda x: __import__("jax").numpy.repeat(x[None, ...], 8, axis=0),
        states,
    )
    img = snapshot_grid(env, states, tile=8)
    w, h = img.size
    assert w == h


def test_snapshot_grid_tile_size_matches_tile_param():
    env = UR5eMJMJX()
    # 16 envs, request tile=4 -> grid 4x4
    n = 16
    states = env.reset()
    states = __import__("jax").tree.map(
        lambda x: __import__("jax").numpy.repeat(x[None, ...], n, axis=0),
        states,
    )
    cell = 48
    img = snapshot_grid(env, states, tile=4, cell=cell)
    expected = 4 * cell + 5 * 1
    assert img.size == (expected, expected)


def test_snapshot_grid_pixel_values_not_all_zero():
    env = UR5eMJMJX()
    states = env.reset()
    states = __import__("jax").tree.map(
        lambda x: __import__("jax").numpy.repeat(x[None, ...], 4, axis=0),
        states,
    )
    img = snapshot_grid(env, states, tile=2)
    arr = np.asarray(img)
    assert arr.shape[-1] == 3
    assert arr.std() > 1.0, "rendered grid should not be a flat color"


def test_snapshot_grid_writes_file(tmp_path):
    env = UR5eMJMJX()
    states = env.reset()
    states = __import__("jax").tree.map(
        lambda x: __import__("jax").numpy.repeat(x[None, ...], 4, axis=0),
        states,
    )
    out = tmp_path / "grid.png"
    img = snapshot_grid(env, states, tile=2)
    img.save(out)
    assert out.exists()
    assert out.stat().st_size > 100
