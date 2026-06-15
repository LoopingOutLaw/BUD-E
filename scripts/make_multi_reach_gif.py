"""Animated GIF of 16 parallel arms performing a scripted REACH task.

Shows that MJX parallelism doesn't just run random noise -- all 16 envs
converge on the same target via a joint-space PD controller.

Outputs:
    demos/multi_arm_reach.gif  - 4x4 grid, 80 frames
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from bude_vla.envs.so101_mjx import UR5eMJMJX
from bude_vla.data.multi_snapshot import snapshot_grid


OUT_DIR = Path(__file__).resolve().parent.parent / "demos"
N_ENVS = 16
N_FRAMES = 80
CELL = 64
TILE = 4


def _pd_reach(qpos, target, kp=8.0, kd=2.0):
    """Simple joint-space PD controller toward a target arm config."""
    err = target - qpos[:, 7:13]
    vel = jnp.zeros_like(err)
    return kp * err - kd * vel


def main():
    out_path = OUT_DIR / "multi_arm_reach.gif"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env = UR5eMJMJX()
    s = env.reset()
    s = jax.tree.map(lambda x: jnp.repeat(x[None, ...], N_ENVS, axis=0), s)

    @jax.jit
    @jax.vmap
    def step(d, a):
        d = d.replace(ctrl=a)
        return mjx.step(env.model, d)

    target = jnp.array([0.0, -0.6, 1.1, 0.0, -0.3, 0.0], dtype=jnp.float32)
    key = jax.random.PRNGKey(1)
    frames = []

    for t in range(N_FRAMES):
        qpos = s.qpos
        action = _pd_reach(qpos, target)
        action = jnp.clip(action, -1.0, 1.0)
        gripper = jnp.zeros((N_ENVS, 1), dtype=jnp.float32)
        full_action = jnp.concatenate([action, gripper], axis=-1)

        s = step(s, full_action)

        if t % 2 == 0:
            img = snapshot_grid(env, s, tile=TILE, cell=CELL)
            frames.append(np.asarray(img))

        if t % 20 == 0:
            print(f"  frame {t}/{N_FRAMES}")

    imageio.mimsave(str(out_path), frames, fps=10, loop=0)
    print(f"wrote {out_path}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
