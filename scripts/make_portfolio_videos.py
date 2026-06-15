"""Generate hi-res portfolio MP4 videos for BUD-E.

Outputs in demos/videos/:
  hero_{cam}.mp4          — single arm, 640x640, 90 frames random actions
  multi_random_{cam}.mp4  — 3x3 grid, cell=256, 60 frames random
  multi_reach_{cam}.mp4   — 3x3 grid, cell=256, 80 frames PD reach
  before_after_{cam}.mp4  — side-by-side 3x3, 80 frames
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mujoco import mjx

from bude_vla.envs.so101_mjx import UR5eMJMJX
from bude_vla.data.multi_snapshot import snapshot_grid, write_video


OUT = Path(__file__).resolve().parent.parent / "demos" / "videos"
OUT.mkdir(parents=True, exist_ok=True)

CAMS = ["over_shoulder", "pov", "front_top", "wrist"]


def _font(size: int = 14):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except Exception:
        return ImageFont.load_default()


def _vmap_step(env):
    @jax.jit
    @jax.vmap
    def step(d, a):
        d = d.replace(ctrl=a)
        return mjx.step(env.model, d)
    return step


def _overlay_text(img: Image.Image, lines: list[str]) -> Image.Image:
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font(max(14, img.size[0] // 60))
    pad = 8
    line_h = font.size + 4
    draw.rectangle([(0, 0), (img.size[0], line_h * len(lines) + pad * 2)], fill=(0, 0, 0))
    for i, ln in enumerate(lines):
        draw.text((pad, pad + i * line_h), ln, fill=(255, 220, 0), font=font)
    return img


def make_hero_videos(env):
    print("Hero videos (single arm, 640x640)...")
    N = 1
    step = _vmap_step(env)
    s = env.reset()
    s = jax.tree.map(lambda x: jnp.repeat(x[None, ...], N, axis=0), s)
    key = jax.random.PRNGKey(7)

    for cam in CAMS:
        frames = []
        sc = s
        key_seq = jax.random.PRNGKey(7)
        for t in range(90):
            key_seq, subkey = jax.random.split(key_seq)
            a = jax.random.uniform(subkey, (N, env.model_mj.nu), minval=-0.3, maxval=0.3)
            sc = step(sc, a)
            if t % 2 == 0:
                img = snapshot_grid(env, sc, tile=1, cell=640, camera=cam)
                frames.append(img)
        out = OUT / f"hero_{cam}.mp4"
        write_video(frames, str(out), fps=15)
        print(f"  wrote hero_{cam}.mp4 ({len(frames)} frames)")


def make_multi_random_videos(env):
    print("Multi-arm random videos (3x3 grid, cell=256)...")
    N = 9
    step = _vmap_step(env)
    s = env.reset()
    s = jax.tree.map(lambda x: jnp.repeat(x[None, ...], N, axis=0), s)

    for cam in CAMS:
        frames = []
        sc = s
        key_seq = jax.random.PRNGKey(42)
        for t in range(60):
            key_seq, subkey = jax.random.split(key_seq)
            a = jax.random.uniform(subkey, (N, env.model_mj.nu), minval=-0.4, maxval=0.4)
            sc = step(sc, a)
            if t % 2 == 0:
                img = snapshot_grid(env, sc, tile=3, cell=256, camera=cam)
                frames.append(img)
        out = OUT / f"multi_random_{cam}.mp4"
        write_video(frames, str(out), fps=15)
        print(f"  wrote multi_random_{cam}.mp4 ({len(frames)} frames)")


def make_multi_reach_videos(env):
    print("Multi-arm PD reach videos (3x3 grid, cell=256)...")
    N = 9
    step = _vmap_step(env)
    s = env.reset()
    s = jax.tree.map(lambda x: jnp.repeat(x[None, ...], N, axis=0), s)
    target = jnp.array([0.0, -0.6, 1.1, 0.0, -0.3, 0.0], dtype=jnp.float32)

    for cam in CAMS:
        frames = []
        sc = jax.tree.map(lambda x: x.copy(), s)
        for t in range(80):
            err = target - sc.qpos[:, 7:13]
            action = 8.0 * err - 2.0 * jnp.zeros_like(err)
            action = jnp.clip(action, -1.0, 1.0)
            gripper = jnp.zeros((N, 1), dtype=jnp.float32)
            full = jnp.concatenate([action, gripper], axis=-1)
            sc = step(sc, full)
            if t % 2 == 0:
                img = snapshot_grid(env, sc, tile=3, cell=256, camera=cam)
                frames.append(img)
        out = OUT / f"multi_reach_{cam}.mp4"
        write_video(frames, str(out), fps=15)
        print(f"  wrote multi_reach_{cam}.mp4 ({len(frames)} frames)")


def make_before_after_videos(env):
    print("Before/after videos (3x3 side-by-side)...")
    N = 9
    step = _vmap_step(env)
    s = env.reset()
    s = jax.tree.map(lambda x: jnp.repeat(x[None, ...], N, axis=0), s)
    target = jnp.array([0.0, -0.6, 1.1, 0.0, -0.3, 0.0], dtype=jnp.float32)
    key = jax.random.PRNGKey(42)
    cell = 200

    for cam in CAMS:
        s_rand = jax.tree.map(lambda x: x.copy(), s)
        s_reach = jax.tree.map(lambda x: x.copy(), s)
        frames = []
        key_seq = key
        for t in range(80):
            key_seq, subkey = jax.random.split(key_seq)
            a_rand = jax.random.uniform(subkey, (N, env.model_mj.nu), minval=-0.5, maxval=0.5)
            s_rand = step(s_rand, a_rand)

            err = target - s_reach.qpos[:, 7:13]
            a_reach = jnp.clip(8.0 * err, -1.0, 1.0)
            gripper = jnp.zeros((N, 1), dtype=jnp.float32)
            a_reach_full = jnp.concatenate([a_reach, gripper], axis=-1)
            s_reach = step(s_reach, a_reach_full)

            if t % 2 == 0:
                left = snapshot_grid(env, s_rand, tile=3, cell=cell, camera=cam)
                right = snapshot_grid(env, s_reach, tile=3, cell=cell, camera=cam)
                w, h = left.size
                combined = Image.new("RGB", (w * 2 + 4, h))
                combined.paste(left, (0, 0))
                combined.paste(right, (w + 4, 0))
                draw = ImageDraw.Draw(combined)
                mid = combined.size[0] // 2
                draw.rectangle([(0, 0), (mid - 2, 22)], fill=(180, 40, 40))
                draw.rectangle([(mid + 2, 0), (combined.size[0], 22)], fill=(40, 140, 40))
                font = _font(13)
                draw.text((6, 3), "Untrained (random)", fill=(255, 255, 255), font=font)
                draw.text((mid + 6, 3), "Trained (scripted reach)", fill=(255, 255, 255), font=font)
                draw.line([(mid, 0), (mid, h)], fill=(200, 200, 200), width=2)
                frames.append(combined)
        out = OUT / f"before_after_{cam}.mp4"
        write_video(frames, str(out), fps=15)
        print(f"  wrote before_after_{cam}.mp4 ({len(frames)} frames)")


def make_task_demo_videos(env):
    """Single-arm task demonstration videos at 640x640 using the scripted policies."""
    from bude_vla.data.demo_recorder import collect_reach_episode, collect_push_episode

    print("Task demo videos (640x640, scripted policies)...")
    import mujoco as _mj

    for cam in CAMS:
        # Reach
        ep = collect_reach_episode(env, np.array([0.6, 0.0, 0.55], dtype=np.float32), n_steps=40)
        frames = []
        d = _mj.MjData(env.model_mj)
        for t in range(ep["qpos"].shape[0]):
            d.qpos[:] = ep["qpos"][t]
            _mj.mj_forward(env.model_mj, d)
            r = _mj.Renderer(env.model_mj, height=640, width=640)
            r.update_scene(d, camera=cam)
            frames.append(Image.fromarray(r.render()))
            r.close()
        out = OUT / f"reach_{cam}.mp4"
        write_video(frames, str(out), fps=20)
        print(f"  wrote reach_{cam}.mp4")

        # Push
        ep = collect_push_episode(env, np.array([0.25, 0.0], dtype=np.float32), n_steps=50)
        frames = []
        d = _mj.MjData(env.model_mj)
        for t in range(ep["qpos"].shape[0]):
            d.qpos[:] = ep["qpos"][t]
            _mj.mj_forward(env.model_mj, d)
            r = _mj.Renderer(env.model_mj, height=640, width=640)
            r.update_scene(d, camera=cam)
            frames.append(Image.fromarray(r.render()))
            r.close()
        out = OUT / f"push_{cam}.mp4"
        write_video(frames, str(out), fps=20)
        print(f"  wrote push_{cam}.mp4")


def main():
    env = UR5eMJMJX()
    make_hero_videos(env)
    make_multi_random_videos(env)
    make_multi_reach_videos(env)
    make_before_after_videos(env)
    make_task_demo_videos(env)
    print(f"\nDone! All videos in {OUT}/")


if __name__ == "__main__":
    main()
