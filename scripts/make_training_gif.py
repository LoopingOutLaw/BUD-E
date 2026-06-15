"""Animated GIF of 64 parallel arms evolving under scripted/random actions.

This produces a visually compelling portfolio piece showing MJX parallelism:
    demos/multi_arm_training.gif  - 4x4 grid animated over 60 steps
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
N_FRAMES = 60
CELL = 64
TILE = 4


def main():
    out_path = OUT_DIR / "multi_arm_training.gif"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env = UR5eMJMJX()
    s = env.reset()
    s = jax.tree.map(lambda x: jnp.repeat(x[None, ...], N_ENVS, axis=0), s)

    @jax.jit
    @jax.vmap
    def step(d, a):
        d = d.replace(ctrl=a)
        return mjx.step(env.model, d)

    key = jax.random.PRNGKey(0)
    frames = []

    for t in range(N_FRAMES):
        key, subkey = jax.random.split(key)
        action = jax.random.uniform(
            subkey, (N_ENVS, env.model_mj.nu), minval=-0.5, maxval=0.5
        )
        s = step(s, action)

        img = snapshot_grid(env, s, tile=TILE, cell=CELL)
        frames.append(np.asarray(img))

        if t % 10 == 0:
            print(f"  frame {t}/{N_FRAMES}")

    imageio.mimsave(str(out_path), frames, fps=12, loop=0)
    print(f"wrote {out_path}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
