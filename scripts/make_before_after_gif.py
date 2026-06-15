"""Before / After comparison GIF.

Left side: 4 arms driven by random actions (simulates untrained VLA).
Right side: 4 arms driven by scripted reach policy (simulates trained VLA).

This is the single most compelling portfolio visual.

Output:
    demos/before_after.gif
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mujoco import mjx

from bude_vla.envs.so101_mjx import UR5eMJMJX
from bude_vla.data.multi_snapshot import snapshot_grid


OUT_DIR = Path(__file__).resolve().parent.parent / "demos"
N_ENVS = 4
N_FRAMES = 80
CELL = 128
TILE = 2


def _add_label(img: Image.Image, left_text: str, right_text: str) -> Image.Image:
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except Exception:
        font = ImageFont.load_default()
    mid = img.size[0] // 2
    bar_h = 22
    draw.rectangle([(0, 0), (mid - 2, bar_h)], fill=(180, 40, 40))
    draw.rectangle([(mid + 2, 0), (img.size[0], bar_h)], fill=(40, 140, 40))
    draw.text((6, 3), left_text, fill=(255, 255, 255), font=font)
    draw.text((mid + 6, 3), right_text, fill=(255, 255, 255), font=font)
    draw.line([(mid, 0), (mid, img.size[1])], fill=(200, 200, 200), width=2)
    return img


def main():
    out_path = OUT_DIR / "before_after.gif"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env = UR5eMJMJX()

    s_rand = env.reset()
    s_rand = jax.tree.map(lambda x: jnp.repeat(x[None, ...], N_ENVS, axis=0), s_rand)
    s_reach = jax.tree.map(lambda x: x.copy(), s_rand)

    @jax.jit
    @jax.vmap
    def step(d, a):
        d = d.replace(ctrl=a)
        return mjx.step(env.model, d)

    def _pd_reach(qpos, target, kp=8.0, kd=2.0):
        err = target - qpos[:, 7:13]
        return kp * err - kd * jnp.zeros_like(err)

    target = jnp.array([0.0, -0.6, 1.1, 0.0, -0.3, 0.0], dtype=jnp.float32)
    key = jax.random.PRNGKey(42)

    frames = []
    for t in range(N_FRAMES):
        # Random side
        key, subkey = jax.random.split(key)
        a_rand = jax.random.uniform(
            subkey, (N_ENVS, env.model_mj.nu), minval=-0.5, maxval=0.5
        )
        s_rand = step(s_rand, a_rand)

        # Scripted reach side
        a_reach = _pd_reach(s_reach.qpos, target)
        a_reach = jnp.clip(a_reach, -1.0, 1.0)
        gripper = jnp.zeros((N_ENVS, 1), dtype=jnp.float32)
        a_reach_full = jnp.concatenate([a_reach, gripper], axis=-1)
        s_reach = step(s_reach, a_reach_full)

        if t % 2 == 0:
            img_rand = snapshot_grid(env, s_rand, tile=TILE, cell=CELL)
            img_reach = snapshot_grid(env, s_reach, tile=TILE, cell=CELL)

            w_img = img_rand.size[0]
            h_img = img_rand.size[1]
            combined = Image.new("RGB", (w_img * 2 + 4, h_img))
            combined.paste(img_rand, (0, 0))
            combined.paste(img_reach, (w_img + 4, 0))
            combined = _add_label(combined, "Untrained (random)", "Trained (scripted reach)")
            frames.append(np.asarray(combined))

        if t % 20 == 0:
            print(f"  frame {t}/{N_FRAMES}")

    imageio.mimsave(str(out_path), frames, fps=10, loop=0)
    print(f"wrote {out_path}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
