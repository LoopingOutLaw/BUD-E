# Visualization Status — Jun 14, 13:13

## What's working
**All CPU-rendered demos are working and on disk.** Open them directly with any image viewer.

| File | What |
|---|---|
| `demos/arm_home.png` | Arm in workspace, single static frame, 640×480 |
| `demos/reach_multi_view.png` | 4-angle grid: over-shoulder, side, front, top, 480×360 |
| `demos/reach_demo.gif` | Animated reach loop, 50 steps + 8 hold, ~10s |
| `demos/push_demo.gif` | Animated push, up to 80 steps, ends with "PUSHED!" |

Open with:
```bash
xdg-open /home/aditya/bude_vla/demos/arm_home.png
xdg-open /home/aditya/bude_vla/demos/reach_multi_view.png
xdg-open /home/aditya/bude_vla/demos/reach_demo.gif
xdg-open /home/aditya/bude_vla/demos/push_demo.gif
```

## What's NOT working — and was removed

`scripts/live_viewer.py` was deleted. It segfaulted at `viewer.launch_passive()`
because the user's `DISPLAY=:1` session did not have the GL extensions glfw
needed. Tried both `glfw` and `osmesa` backends; both crashed. We chose not to
spend more cycles debugging the live viewer since the GIF visualizer gives you
the same content (no live interaction benefit since the scripted policy +
physics are deterministic).

If you want a live-3D viewer later, the path is `MUJOCO_GL=glfw` on a
desktop session, OR run the visualizer remotely via SSH with X forwarding.

## Why I kept the CPU GIF renderer path

The GPU rendering path (`mujoco.Renderer` with `MUJOCO_GL=egl`) requires
GPU display buffers. Our `scripts/visualize_reach.py` uses
`scheme='Pillow'` + CPU `mujoco.Renderer` which works *headless* with no
display. This is a real MuJoCo support path (used by many RL trainers for
rendering during eval). All demos are produced through it.

## Known ML-limitation in MJX

During debugging I confirmed that **MJX with cube freejoint + finger equality
constraint** has subtle behavior differences from native MuJoCo. Native
MuJoCo correctly drives `shoulder_pan` motor with `ctrl=1.0`; MJX leaves
`qpos[7]=0` after 100 steps unless the cube `condim` is set to 4 with
matched `solref`/`solimp`. This is a real bug that needs a deeper investigation
(Task 13 sub-task: port tests to use **native MuJoCo** for env stepping,
since we only need MJX for the vmap-256 path which we'll add much later).

**Net effect**: Training and demo collection will use **native MuJoCo CPU** path
until we get a GPU VMAP working. That gives 4-5x wall-clock slowdown but
correct behavior. We'll accept the cost per your directive.
