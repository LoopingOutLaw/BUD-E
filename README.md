# BUD-E

BUD-E is a compact Vision-Language-Action training stack for an SO-101 / LeRobot-style 6-DoF arm in MuJoCo. The current active target is red-cube pick-and-place from dual-camera images, proprioception, and a language instruction.

The policy is intentionally not given privileged cube coordinates. Scripted demonstrators may use simulator state to generate successful examples, but the learned policy runs from visual observations and robot-side proprio/contact signals.

## Current Status

The current best base run is `pick_v26_unified`:

- DINOv2 vision at 224 px.
- Dual-camera observations with 4-frame history.
- 9D proprioception without progress/time shortcut leakage.
- Context/perception-conditioned action decoding.
- Gripper-weighted BC and flow losses.
- Recovery-jitter and touch/nudge backoff demonstration data.
- Smoothed closed-loop eval through temporal ensembling.

Observed behavior after v26: the arm now reaches toward the cube almost every rollout. The remaining failure mode is precision near the final centimeters: small XY offset and occasional early gripper close. The current next step is `pick_v27_precision`, a precision fine-tune from the v26 60k checkpoint.

Detailed experiment notes and exact reproduction commands are in [`docs/pick_vla_training_notes.md`](docs/pick_vla_training_notes.md).

## Repository Contents

```text
scripts/record_pick_episodes.py   Record scripted pick demonstrations
scripts/build_frame_cache.py      Build cached image/history training frames
scripts/train.py                  Train or fine-tune the VLA policy
scripts/eval_pick_ball.py         Closed-loop MuJoCo eval and MP4 rendering
scripts/inspect_ckpt.py           Inspect checkpoint architecture/config
src/bude_vla/                     Core data, env, model, perception, rollout code
tests/                            Focused training-control tests
urdf/                             SO-101 model and pick scene assets
```

Generated datasets, frame caches, videos, logs, and checkpoints are ignored by Git. Keep them local unless they are deliberately published as release artifacts.

## Environment

The commands below assume the local venv used during development:

```bash
cd /home/aditya/bude_vla
PYTHONPATH=src /home/aditya/venv-bude/bin/python -m pip install -e ".[dev,sim]"
```

For MuJoCo rendering on the RTX 4060 laptop, use:

```bash
export MUJOCO_GL=egl
export PYTHONPATH=src
```

## Current v27 Overnight Run

Record precision data, build a cache, then fine-tune from v26:

```bash
cd /home/aditya/bude_vla
mkdir -p logs

MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/record_pick_episodes.py \
  --max-eps 7000 \
  --out /home/aditya/bude_vla/data/pick_v27_precision \
  --seed 77 \
  --img-size 224 \
  --recovery-jitter-xy 0.004 \
  --recovery-jitter-z 0.010 \
  --recovery-jitter-prob 0.45 \
  --nudge-recovery-prob 0.35 \
  --nudge-recovery-xy 0.010 \
  --nudge-recovery-z 0.010 \
  --max-grasp-retries 0 \
&& PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/build_frame_cache.py \
  --data-root /home/aditya/bude_vla/data/pick_v27_precision \
  --out-dir /home/aditya/bude_vla/data/pick_v27_precision/cache_224_h4_phase48k \
  --max-frames 48000 \
  --n-history-frames 4 \
  --phase-bins 8 \
  --seed 77 \
&& PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/train.py \
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
  --ema-decay 0.999 2>&1 | tee logs/pick_v27_precision.log
```

Watch progress:

```bash
tail -f /home/aditya/bude_vla/logs/pick_v27_precision.log
```

## Evaluation

Use smoothed eval settings. They reduce jitter by trusting queued actions more and replanning every two control steps:

```bash
cd /home/aditya/bude_vla

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

## Development Notes

- Keep integrated train-time eval disabled on the laptop with `--eval-every 0`; manual eval avoids RAM spikes.
- Use `--num-workers 0` for safer overnight training on 16 GB RAM.
- Remove old frame caches and intermediate checkpoints when storage gets tight.
- If a data recording command is accidentally started twice, delete the partial dataset before restarting.
- Prefer precision data and gripper/late-phase weighting for early-close failures; do not add privileged cube-position inputs to the learned policy.

## Inspiration

Built with reference to small VLA and robot imitation-learning ideas from X-VLA, SmolVLA, Octo, LeRobot-style datasets, flow matching, and ACT-style temporal ensembling.

## Citation

Built by Aditya, 2026.
