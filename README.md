# BUD-E

BUD-E is a compact Vision-Language-Action training stack for an SO-101 / LeRobot-style 6-DoF arm in MuJoCo. The active target is red-cube pick-and-place from dual-camera images, proprioception, contact state, and a language instruction.

The learned policy is intentionally not given privileged cube coordinates. Scripted demonstrators may use simulator state to generate successful examples, but the policy runs from visual observations and robot-side proprio/contact signals only.

## Current Status

The current best line is `pick_v29_mixed_reach_precision_recovery` followed by the focused `pick_v30_contact_timing` fine-tune.

What is working now:

- DINOv2 vision at 224 px.
- Dual-camera observations with 4-frame history.
- 9D proprioception without the old progress/time shortcut.
- Context/perception-conditioned action decoding.
- Mixed v26/v27/v28 training to preserve reaching while adding precision and recovery.
- Recovery-jitter, descent-depth, retry, and nudge/backoff demonstration support.
- Contact-focused cache sampling through `scripts/build_frame_cache.py --phase-ranges`.

Observed behavior after v29: the arm reliably moves toward the cube, descends, and in some failures re-attempts after pushing or missing the cube. The remaining failure is terminal grasp control: close timing can be late/early, and the arm can stall near contact. v30 is a focused fine-tune for that contact/close phase.

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

Generated datasets, frame caches, videos, logs, and checkpoints are ignored by Git. Keep them local unless deliberately published as release artifacts.

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

## Current Training Path

The current stable approach is not v28-only training. v28 fit its dataset but regressed rollout reaching, so v29 mixes v26, v27, and v28 while initializing from the strong v26 base. v30 then focuses the cache on descent/contact/close timing.

Build contact-focused caches:

```bash
cd /home/aditya/bude_vla
PHASE_RANGES="0.06:0.20:0.40,0.20:0.42:0.45,0.42:0.70:0.12,0.70:1.00:0.03"

MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/build_frame_cache.py \
  --data-root data/pick_v26_unified \
  --out-dir data/pick_v26_unified/cache_224_h4_contact8k \
  --max-frames 8000 \
  --n-history-frames 4 \
  --phase-ranges "$PHASE_RANGES"
```

Repeat the same cache command for `pick_v27_precision` and `pick_v28_depth_nudge_recovery`, changing only `--data-root` and `--out-dir`.

Fine-tune v30 from v29:

```bash
FRAME_CACHE="data/pick_v26_unified/cache_224_h4_contact8k:data/pick_v27_precision/cache_224_h4_contact8k:data/pick_v28_depth_nudge_recovery/cache_224_h4_contact8k"

MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/train.py \
  --data-root data/pick_v26_unified \
  --data-root data/pick_v27_precision \
  --data-root data/pick_v28_depth_nudge_recovery \
  --frame-cache "$FRAME_CACHE" \
  --task pick_v30_contact_timing \
  --init-from checkpoints/pick_v29_mixed_reach_precision_recovery/pick_v29_mixed_reach_precision_recovery_final.pt \
  --use-dinov2 \
  --img-size 224 \
  --chunk-size 16 \
  --n-history-frames 4 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --num-workers 2 \
  --n-steps 25000 \
  --save-every 5000 \
  --eval-every 0 \
  --lr 2e-5 \
  --backbone-lr 1e-6 \
  --bc-loss-weight 7.0 \
  --flow-loss-weight 0.10 \
  --gripper-loss-weight 12.0 \
  --early-bc-weight 3.0 \
  --early-bc-frac 0.10 \
  --late-bc-weight 18.0 \
  --late-bc-frac 0.18 \
  --ema-decay 0.999
```

## Evaluation

Use fixed cube positions so v29/v30 videos are comparable:

```bash
cd /home/aditya/bude_vla

MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/eval_pick_ball.py \
  --ckpt checkpoints/pick_v30_contact_timing/pick_v30_contact_timing_final.pt \
  --num-episodes 8 \
  --max-steps 1800 \
  --exec-first-only \
  --cube-positions '0.25,0.00;0.30,-0.04;0.30,0.06;0.22,0.05;0.28,-0.08;0.34,0.03;0.18,-0.04;0.31,0.08' \
  --out demos/videos/eval_pick_v30_final_firstonly.mp4
```

For smoother rollout:

```bash
MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/eval_pick_ball.py \
  --ckpt checkpoints/pick_v30_contact_timing/pick_v30_contact_timing_final.pt \
  --num-episodes 8 \
  --max-steps 1800 \
  --ensembling \
  --ensembling-k 0.55 \
  --replan-every 1 \
  --cube-positions '0.25,0.00;0.30,-0.04;0.30,0.06;0.22,0.05;0.28,-0.08;0.34,0.03;0.18,-0.04;0.31,0.08' \
  --out demos/videos/eval_pick_v30_final_ensemble.mp4
```

If the arm reaches but fails to grasp, run `--debug-actions` and inspect the printed `grip=` values around contact. A late transition from positive/open to negative/closed indicates close timing is still late; a negative command with no grasp points to depth/contact alignment.

## Development Notes

- Keep integrated train-time eval disabled on the laptop with `--eval-every 0`; manual eval avoids RAM spikes.
- Use cached frames for training. Lazy MP4 decode is correct but much slower.
- Remove old frame caches and intermediate checkpoints when storage gets tight; raw datasets and final checkpoints are more valuable.
- If a data recording command is accidentally started twice, delete the partial dataset before restarting.
- Prefer visual/contact recovery data and loss weighting over privileged cube-position inputs.

## Inspiration

Built with reference to small VLA and robot imitation-learning ideas from X-VLA, SmolVLA, Octo, LeRobot-style datasets, flow matching, and ACT-style temporal ensembling.

## Citation

Built by Aditya, 2026.
