# 🎬 First Visual Milestone — Jun 14

**Two rendered demos available** in `demos/`:

| File | What |
|---|---|
| `arm_home.png` (35KB) | Static arm in workspace, single PNG, 640×480 |
| `reach_demo.gif` (190KB) | Scripted reach — arm moves to fixed target, 50 steps + 8 hold, 15fps |
| `push_demo.gif` (257KB) | Scripted push — arm pushes red cube toward cyan target zone |

## How to regenerate
```bash
cd /home/aditya/bude_vla && PYTHONPATH=src /home/aditya/.bude-venv/bin/python \
    scripts/visualize_reach.py --mode static
cd /home/aditya/bude_vla && PYTHONPATH=src /home/aditya/.bude-venv/bin/python \
    scripts/visualize_reach.py --mode reach --steps 50 --out demos/reach_demo.gif
cd /home/aditya/bude_vla && PYTHONPATH=src /home/aditya/.bude-venv/bin/python \
    scripts/visualize_reach.py --mode push  --steps 80 --out demos/push_demo.gif
```

## Why this is a checkpoint
You can **see** the arm in a real MuJoCo scene rendering the actual XML. This validates:
- XML parses correctly
- Cube body sits at the right place
- MuJoCo forward-dynamics works in headless CPU mode
- Renderer's projection + textures render correctly through `mujoco.mj_forward`
- Scripted policies drive the arm in physically meaningful motion

## What was hard-won
The hang that plagued `tests/envs/test_lerobot_v3::test_two_episodes_load_correctly`
and `tests/data/test_push::test_push_episode_runs` is **solved by sidestep**, not by root
cause: single-Renderer-per-script prevents it. The real hang is likely `mujoco.Renderer`
state interaction with `mjx.get_data` round-tripping — out of scope for the visualizer
path which only uses raw `mujoco.MjData`.

## Next
- Real recording pipeline (Task 13) — needs the hang *fixed*, not sidestepped
- True training loop
- Eval + GIF visualizer with the **trained** policy
