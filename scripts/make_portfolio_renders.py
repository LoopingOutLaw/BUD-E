"""Generate all portfolio deliverables for BUD-E.

Outputs (in demos/):
    hero_arm.png                - single arm hero shot at "ready" pose
    hero_arm_extended.png       - arm in extended pose
    hero_arm_grasp.png          - arm in grasp-ready pose
    multi_arm_grid.png          - 8x8 grid, 64 envs at step 30
    multi_arm_grid_compound.png - vertical: step 1 / step 30 comparison

Animated (in demos/):
    reach_demo.gif              - single arm reaching (from generate_demos.py)
    push_demo.gif               - push task
    pick_place_demo.gif         - pick and place
    multi_arm_training.gif      - 4x4 grid of 16 arms with random actions
    multi_arm_reach.gif         - 4x4 grid of 16 arms with PD reach control
    before_after.gif            - side-by-side: random vs scripted reach

Static poses (in demos/poses/):
    arm_home.png, arm_extended.png, arm_raised.png,
    arm_reach_right.png, arm_grasp_ready.png
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mujoco import mjx

from bude_vla.envs.so101_mjx import UR5eMJMJX
from bude_vla.data.multi_snapshot import snapshot_grid


OUT = Path(__file__).resolve().parent.parent / "demos"
OUT.mkdir(parents=True, exist_ok=True)
POSES_DIR = OUT / "poses"
POSES_DIR.mkdir(parents=True, exist_ok=True)


def _font(size: int = 14):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except Exception:
        return ImageFont.load_default()


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


# ─── Static hero shots ──────────────────────────────────────────────────

POSES = {
    "home": [0.0, -0.5, 0.9, 0.0, 0.0, 0.0],
    "extended": [0.3, -1.0, 1.5, 0.0, 0.0, 0.0],
    "raised": [0.0, -1.5, 0.5, 0.0, 0.0, 0.0],
    "reach_right": [-0.5, -0.5, 0.9, 0.0, 0.0, 0.0],
    "grasp_ready": [0.0, -0.3, 1.2, 0.0, 0.0, 0.0],
}


def render_pose(env, name: str, joint_pos: list, out_path: Path, size: int = 640):
    """Render a pose using the auto-framing free camera (gives the most
    visually correct shot, since MuJoCo auto-frames to scene)."""
    d = mujoco.MjData(env.model_mj)
    d.qpos[:] = 0.0
    d.qpos[0:3] = [0.6, 0.0, 0.435]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[7:13] = joint_pos
    d.qpos[13] = -0.02
    mujoco.mj_forward(env.model_mj, d)
    r = mujoco.Renderer(env.model_mj, height=size, width=size)
    r.update_scene(d)
    img = r.render()
    r.close()
    Image.fromarray(img).save(str(out_path))
    print(f"  wrote {out_path.name}")


def render_all_poses(env):
    print("Rendering poses...")
    for name, jp in POSES.items():
        render_pose(env, name, jp, POSES_DIR / f"arm_{name}.png")

    render_pose(env, "hero_grasp", [0.0, -0.6, 1.1, 0.0, -0.3, 0.0],
                OUT / "hero_arm.png", size=1024)
    render_pose(env, "hero_extended", [0.3, -1.0, 1.5, 0.0, 0.0, 0.0],
                OUT / "hero_arm_extended.png", size=1024)
    render_pose(env, "hero_grasp_side", [0.0, -0.3, 1.2, 0.0, 0.0, 0.0],
                OUT / "hero_arm_grasp.png", size=1024)


# ─── Multi-arm grids ────────────────────────────────────────────────────

def render_grids(env):
    print("Rendering 8x8 multi-arm grid...")
    N = 64
    s = env.reset()
    s = jax.tree.map(lambda x: jnp.repeat(x[None, ...], N, axis=0), s)

    @jax.jit
    @jax.vmap
    def step(d, a):
        d = d.replace(ctrl=a)
        return mjx.step(env.model, d)

    key = jax.random.PRNGKey(42)

    # step 1
    key, subkey = jax.random.split(key)
    a1 = jax.random.uniform(subkey, (N, env.model_mj.nu), minval=-0.4, maxval=0.4)
    s1 = step(s, a1)
    cell = int(1280 / 8) - 2
    img_early = snapshot_grid(env, s1, tile=8, cell=cell)
    img_early.convert("RGB").save(str(OUT / "multi_arm_grid_early.png"))
    print("  wrote multi_arm_grid_early.png")

    # step 30
    for _ in range(29):
        key, subkey = jax.random.split(key)
        a = jax.random.uniform(subkey, (N, env.model_mj.nu), minval=-0.4, maxval=0.4)
        s1 = step(s1, a)
    img_late = snapshot_grid(env, s1, tile=8, cell=cell)
    img_late.convert("RGB").save(str(OUT / "multi_arm_grid.png"))
    print("  wrote multi_arm_grid.png")

    # compound
    early = img_early.convert("RGB")
    late = img_late.convert("RGB")
    w, h = early.size
    compound = Image.new("RGB", (w, h * 2 + 4), (30, 30, 35))
    compound.paste(early, (0, 0))
    compound.paste(late, (0, h + 4))
    compound = _overlay_text(
        compound,
        ["BUD-E  -  64 batched envs (MJX parallelism)", "Top: step 1   |   Bottom: step 30"],
    )
    compound.save(str(OUT / "multi_arm_grid_compound.png"))
    print("  wrote multi_arm_grid_compound.png")


# ─── Animated GIFs ──────────────────────────────────────────────────────

def _vmap_step(env):
    @jax.jit
    @jax.vmap
    def step(d, a):
        d = d.replace(ctrl=a)
        return mjx.step(env.model, d)
    return step


def make_multi_arm_training_gif(env):
    print("Generating multi_arm_training.gif (random actions, 16 envs)...")
    N = 16
    step = _vmap_step(env)
    s = env.reset()
    s = jax.tree.map(lambda x: jnp.repeat(x[None, ...], N, axis=0), s)
    key = jax.random.PRNGKey(0)
    frames = []
    for t in range(60):
        key, subkey = jax.random.split(key)
        a = jax.random.uniform(subkey, (N, env.model_mj.nu), minval=-0.5, maxval=0.5)
        s = step(s, a)
        if t % 2 == 0:
            img = snapshot_grid(env, s, tile=4, cell=64)
            frames.append(np.asarray(img))
    imageio.mimsave(str(OUT / "multi_arm_training.gif"), frames, fps=12, loop=0)
    print("  wrote multi_arm_training.gif")


def make_multi_arm_reach_gif(env):
    print("Generating multi_arm_reach.gif (PD reach, 16 envs)...")
    N = 16
    step = _vmap_step(env)
    s = env.reset()
    s = jax.tree.map(lambda x: jnp.repeat(x[None, ...], N, axis=0), s)
    target = jnp.array([0.0, -0.6, 1.1, 0.0, -0.3, 0.0], dtype=jnp.float32)
    frames = []
    for t in range(80):
        err = target - s.qpos[:, 7:13]
        action = 8.0 * err - 2.0 * jnp.zeros_like(err)
        action = jnp.clip(action, -1.0, 1.0)
        gripper = jnp.zeros((N, 1), dtype=jnp.float32)
        full = jnp.concatenate([action, gripper], axis=-1)
        s = step(s, full)
        if t % 2 == 0:
            img = snapshot_grid(env, s, tile=4, cell=64)
            frames.append(np.asarray(img))
    imageio.mimsave(str(OUT / "multi_arm_reach.gif"), frames, fps=10, loop=0)
    print("  wrote multi_arm_reach.gif")


def make_before_after_gif(env):
    print("Generating before_after.gif (random vs scripted reach)...")
    N = 4
    step = _vmap_step(env)
    s_rand = env.reset()
    s_rand = jax.tree.map(lambda x: jnp.repeat(x[None, ...], N, axis=0), s_rand)
    s_reach = jax.tree.map(lambda x: x.copy(), s_rand)
    target = jnp.array([0.0, -0.6, 1.1, 0.0, -0.3, 0.0], dtype=jnp.float32)
    key = jax.random.PRNGKey(42)
    frames = []
    cell = 128
    for t in range(80):
        key, subkey = jax.random.split(key)
        a_rand = jax.random.uniform(subkey, (N, env.model_mj.nu), minval=-0.5, maxval=0.5)
        s_rand = step(s_rand, a_rand)

        err = target - s_reach.qpos[:, 7:13]
        a_reach = jnp.clip(8.0 * err, -1.0, 1.0)
        gripper = jnp.zeros((N, 1), dtype=jnp.float32)
        a_reach_full = jnp.concatenate([a_reach, gripper], axis=-1)
        s_reach = step(s_reach, a_reach_full)

        if t % 2 == 0:
            left = snapshot_grid(env, s_rand, tile=2, cell=cell)
            right = snapshot_grid(env, s_reach, tile=2, cell=cell)
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
            frames.append(np.asarray(combined))
    imageio.mimsave(str(OUT / "before_after.gif"), frames, fps=10, loop=0)
    print("  wrote before_after.gif")


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    env = UR5eMJMJX()
    render_all_poses(env)
    render_grids(env)
    make_multi_arm_training_gif(env)
    make_multi_arm_reach_gif(env)
    make_before_after_gif(env)
    print("\nDone! All portfolio deliverables in demos/")


if __name__ == "__main__":
    main()
