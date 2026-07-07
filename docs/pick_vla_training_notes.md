# Pick VLA Training Notes

This note records the current working direction for the SO-101 pick-and-place VLA experiments. It is intentionally practical: what failed, what was changed, what is working now, and which commands reproduce the current training path.

## Core Constraint

The policy must not receive the cube position as an oracle input. The scripted demonstrator may use simulator state to produce demonstrations, but the learned policy observes only:

- dual camera images,
- robot proprioception,
- language instruction,
- contact/proprio signals already available to the robot-side policy interface.

This keeps the setup aligned with eventual real-time robot deployment, where the arm will not be handed privileged cube coordinates.

## What Failed

Early runs could train for many steps while still producing 0 percent closed-loop success. The recurring rollout symptoms were:

- moving to a fixed pose regardless of cube position,
- touching or hovering near the cube without grasping,
- closing too early and pushing the cube away,
- freezing after a near miss.

The major causes found during debugging were:

- insufficient recovery behavior in success-only demonstrations,
- weak gripper supervision relative to arm joint dimensions,
- too much reliance on shortcut-like progress/time information,
- retry/miss data overpowering clean grasp data when mixed too aggressively,
- jittery evaluation when replanning every step with low temporal smoothing.

## Important Code Changes

The current code path includes these fixes:

- DINOv2 vision at 224 px with dual camera history stacking.
- Visual/perception-conditioned action decoding so actions cannot collapse to proprio-only behavior.
- Recovery-jitter demonstrations in `scripted_pick_and_place.py` and `record_pick_episodes.py`.
- Optional retry demonstrations, used carefully and not in the precision fine-tune.
- Mixed cached training support for multiple roots.
- Removal of the progress-proprio shortcut after it caused rollout timing leakage.
- Gripper-weighted BC and flow losses so close/open timing matters during training.
- ACT-style temporal ensembling at eval time with tunable smoothing.

## Current Best Result

The best base run is `pick_v26_unified` from a fresh unified dataset. It changed the behavior qualitatively:

- the arm now visually reaches toward the cube almost every rollout,
- fixed-pose collapse is gone,
- inference smoothing removes most shaky motion,
- the remaining issue is final centimeter-level offset and early close timing.

The useful preserved base checkpoint is:

```text
checkpoints/pick_v26_unified/pick_v26_unified_step_060000.pt
```

Generated datasets, frame caches, videos, logs, and checkpoints are intentionally ignored by Git.

## v26 Base Dataset

The v26 dataset was recorded with a mix of clean, recovery-jitter, and light retry behavior:

```bash
MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/record_pick_episodes.py \
  --max-eps 4000 \
  --out /home/aditya/bude_vla/data/pick_v26_unified \
  --seed 60 \
  --img-size 224 \
  --recovery-jitter-xy 0.008 \
  --recovery-jitter-prob 0.65 \
  --max-grasp-retries 1 \
  --retry-miss-xy 0.008 \
  --retry-miss-prob 0.20
```

The matching v26 training run used a fresh chunk-16 architecture:

```bash
PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/train.py \
  --data-root /home/aditya/bude_vla/data/pick_v26_unified \
  --frame-cache /home/aditya/bude_vla/data/pick_v26_unified/cache_224_h4_phase32k \
  --task pick_v26_unified \
  --use-dinov2 \
  --img-size 224 \
  --chunk-size 16 \
  --n-history-frames 4 \
  --batch-size 2 \
  --grad-accum-steps 16 \
  --num-workers 0 \
  --n-steps 60000 \
  --save-every 10000 \
  --eval-every 0 \
  --lr 3e-5 \
  --backbone-lr 1e-6 \
  --bc-loss-weight 4.0 \
  --flow-loss-weight 0.10 \
  --gripper-loss-weight 8.0 \
  --late-bc-weight 10.0 \
  --late-bc-frac 0.45 \
  --early-bc-weight 1.5 \
  --early-bc-frac 0.20 \
  --ema-decay 0.999
```

## v27 Precision Fine-Tune

The next step is not a full reset. It is a precision-grasp fine-tune from v26. The goal is to reduce final XY/Z offset, teach descent-depth correction, and delay gripper closing until the wrist is centered over the cube.

Record cleaner precision and descent-depth recovery data:

```bash
MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/record_pick_episodes.py \
  --max-eps 7000 \
  --out /home/aditya/bude_vla/data/pick_v27_precision \
  --seed 77 \
  --img-size 224 \
  --recovery-jitter-xy 0.004 \
  --recovery-jitter-z 0.010 \
  --recovery-jitter-prob 0.35 \
  --max-grasp-retries 0
```

Build the frame cache:

```bash
PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/build_frame_cache.py \
  --data-root /home/aditya/bude_vla/data/pick_v27_precision \
  --out-dir /home/aditya/bude_vla/data/pick_v27_precision/cache_224_h4_phase48k \
  --max-frames 48000 \
  --n-history-frames 4 \
  --phase-bins 8 \
  --seed 77
```

Fine-tune from the v26 60k checkpoint:

```bash
PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/train.py \
  --data-root /home/aditya/bude_vla/data/pick_v27_precision \
  --frame-cache /home/aditya/bude_vla/data/pick_v27_precision/cache_224_h4_phase48k \
  --task pick_v27_precision \
  --init-from checkpoints/pick_v26_unified/pick_v26_unified_step_060000.pt \
  --use-dinov2 \
  --img-size 224 \
  --chunk-size 16 \
  --n-history-frames 4 \
  --batch-size 2 \
  --grad-accum-steps 16 \
  --num-workers 0 \
  --n-steps 25000 \
  --save-every 5000 \
  --eval-every 0 \
  --lr 1e-5 \
  --backbone-lr 5e-7 \
  --bc-loss-weight 5.0 \
  --flow-loss-weight 0.05 \
  --gripper-loss-weight 10.0 \
  --late-bc-weight 14.0 \
  --late-bc-frac 0.45 \
  --early-bc-weight 2.0 \
  --early-bc-frac 0.20 \
  --ema-decay 0.999
```

## Eval Command

Use the smoothed eval settings. They reduce jitter by trusting queued actions more and replanning every two steps:

```bash
MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/eval_pick_ball.py \
  --ckpt checkpoints/pick_v27_precision/pick_v27_precision_step_020000.pt \
  --num-episodes 8 \
  --max-steps 1800 \
  --ensembling \
  --ensembling-k 0.75 \
  --replan-every 2 \
  --cube-positions '0.25,0.00;0.30,-0.04;0.30,0.06;0.22,0.05;0.28,-0.08;0.34,0.03;0.18,-0.04;0.31,0.08' \
  --out demos/videos/eval_pick_v27_step_020000_smooth.mp4
```

## Operational Notes

- Keep `--eval-every 0` during training on the laptop; integrated eval can spike RAM.
- Use `num-workers 0` with cached frames for the safest overnight runs.
- Keep only the best/latest checkpoints during iteration. Older step checkpoints and frame caches can consume tens of GB.
- If a run is started twice, delete the partial dataset before restarting to avoid mixing inconsistent data.
- If the model reaches the cube but closes early, prefer precision data and gripper/late-phase weighting over more generic retry data.
