# BUD-E

BUD-E is a compact vision-language-action stack for an SO-101 / LeRobot-style
arm in MuJoCo. The active task is to pick a red cube and place it in the blue
target zone from top and wrist RGB, robot joint state, and a language command.

The learned policy is never given simulator cube coordinates. Its deployable
observation contract is camera pixels plus the six measured arm/gripper joint
positions. Simulator object state is used only by demonstration teachers and
post-rollout metrics.

## Current Status

The active experiment is the clean `pick_v37_camera_fixed` reset. Do not mix
v26-v36 data, caches, or checkpoints into this run: those artifacts were made
with a different camera geometry and control-rate contract and have been
removed locally.

Verified on 2026-07-11 before training:

- Camera-only calibration: 0.32 mm median initial cube-localization error.
- Camera-only scripted expert: 50/50 strict grasps and 50/50 successful places.
- Fresh disk replay smoke test: 19/19 strict grasps and 19/19 successful places.
- Test suite: 22/22 passing.
- Free disk after obsolete-artifact cleanup: about 89 GiB.

The 50/50 result validates perception, mechanics, workspace, IK, and task
success logic. It is not a claim that the learned v37 checkpoint already has
100% success. The pipeline below trains that checkpoint from fresh compatible
data and measures it on 150 random positions.

## Run V37

Run one guarded pipeline from the repository root:

```bash
cd /home/aditya/bude_vla
bash scripts/run_v37_camera_fixed.sh
```

There is no timeout. The script runs these stages in order:

1. Require at least 95% camera-only expert success over 100 random episodes.
2. Record 4,000 fresh 224px attempts at the corrected 31.25 Hz policy rate.
3. Require at least 3,200 successful demonstrations.
4. Replay 200 persisted action sequences and require at least 95% success.
5. Build one 6,000-frame, two-history cache sized for a 16 GB RAM laptop.
6. Train from scratch for 25,000 steps and evaluate every 5,000 steps.
7. Preserve the checkpoint with the best closed-loop success, then run a
   150-position benchmark and render an 8-position diagnostic video.

The runner is restart-aware for completed recording/cache/training stages. It
refuses to reuse a partial dataset or cache because silently mixing partial
artifacts is more costly than failing early.

Watch progress from another terminal:

```bash
cd /home/aditya/bude_vla
tail -F \
  logs/pick_v37_visual_expert_bench.log \
  logs/pick_v37_record.log \
  logs/pick_v37_replay.log \
  logs/pick_v37_cache.log \
  logs/pick_v37_train.log \
  logs/pick_v37_random_bench.log
```

The selected checkpoint path is written to
`logs/pick_v37_selected_checkpoint.txt`. The final video is
`demos/videos/eval_pick_v37_camera_fixed.mp4`.

## What Was Actually Broken

The old 0% result was not explained by insufficient training steps. Several
data and inference contracts were wrong at the same time:

- **Camera coverage:** `front_top` was tilted and did not reliably cover the
  old random workspace, especially negative Y positions.
- **False red target:** bright red fingertip debug geometry was visible in the
  policy image. The red centroid could follow the gripper instead of the cube.
- **Action-rate mismatch:** demonstrations ran at 125 Hz while MP4 metadata and
  learned rollout behavior assumed about 30 Hz. Long near-identical action
  runs taught the visible freeze mode.
- **DINO history wiring:** pretrained RGB patch weights were attached to the
  oldest top frame, not the current top frame, with history enabled.
- **DINO preprocessing:** image channels were not normalized as expected by
  the pretrained DINOv2 backbone.
- **Teacher state bugs:** retry displacement was ignored during lift, stale
  one-frame contact could count as a grasp, and lift interpolation jumped
  before reaching its endpoint.
- **Evaluation RAM:** non-video closed-loop evaluation retained every stacked
  frame, producing large RAM spikes alongside multi-gigabyte caches.

Training longer on those artifacts could reduce supervised loss while keeping
rollout success at zero. V37 fixes the contracts first and then trains fresh.

## V37 Policy Contract

Inputs:

- 224x224 top RGB and wrist RGB.
- Two observation times, newest frame last.
- Five arm joint positions plus one gripper position (`state_dim=6`).
- The language command: `pick up the red cube and place it in the blue target zone`.
- A red-component centroid computed from RGB pixels, not world coordinates.

Outputs:

- Six absolute joint/gripper targets.
- Sixteen future actions per prediction.
- One executed action every 16 MuJoCo substeps, matching the recorded
  31.25 Hz stream.
- Closed-loop temporal ensembling replans every action during evaluation.

The policy receives no episode-progress clock, cube pose, target-relative cube
vector, simulator contact bit, or strict-grasp bit. This keeps the active input
layout reproducible on the real arm with cameras and joint encoders.

## Data Rules

- Only successful demonstrations enter v37.
- Approximately 70% are clean trajectories.
- Mild XY/Z recovery, light nudge recovery, and one retry cover local errors
  without dominating the clean task strategy.
- The IK expert plans at 125 Hz and every fourth action target is retained.
- That retained plan is replayed at the exact 31.25 Hz deployment rate.
- Images and proprio are recorded from the replay, so every observation/action
  pair and state transition matches learned-policy execution.
- Plans that fail during policy-rate replay are discarded before writing.
- `meta/info.json` and each episode index persist the actual FPS, record stride,
  simulator substeps per action, and initial cube location for replay testing.
- Training is forbidden until persisted actions solve at least 95% of the
  replay sample.

These checks follow the official OpenVLA troubleshooting advice to replay
demonstrations, verify the inference/data contract, avoid excessive idle
actions, and ensure coverage of test variation:
[OpenVLA troubleshooting](https://github.com/openvla/openvla#vla-performance-troubleshooting).
The use of action chunks and temporal ensembling is also aligned with
[LeRobot's ACT guidance](https://huggingface.co/docs/lerobot/act), while the
repeated random-position coverage follows
[LeRobot's SmolVLA data guidance](https://huggingface.co/docs/lerobot/smolvla).

## Memory And Storage

The v37 cache is approximately 3.4 GiB:

```text
6000 frames * 224 * 224 * (2 histories * 6 RGB channels) * 1 byte
```

Training uses batch size 4, gradient accumulation 8, and one worker. The
effective batch is 32 without loading a 14 GiB cache or several worker copies
into RAM. The runner checks free disk and available RAM before expensive
stages, sets a repository-local temporary directory, and does not retain eval
frames unless video recording is explicitly enabled.

## Repository Map

```text
scripts/run_v37_camera_fixed.sh       Guarded end-to-end v37 pipeline
scripts/benchmark_visual_servo_pick.py Camera-only perception/mechanics gate
scripts/record_pick_episodes.py        Fresh demonstration recorder
scripts/validate_dataset_replay.py     Persisted-action contract gate
scripts/build_frame_cache.py           Bounded frame/history cache builder
scripts/train.py                       VLA training and best-checkpoint eval
scripts/benchmark_random_pick.py       Random-position learned-policy benchmark
scripts/eval_pick_ball.py              Closed-loop video evaluation
src/bude_vla/visual_servo.py           RGB localization and camera calibration
src/bude_vla/                          Model, data, environment, and rollout code
tests/                                 Focused regression tests
```

Detailed root-cause evidence and the decision protocol after v37 are in
[`docs/pick_vla_training_notes.md`](docs/pick_vla_training_notes.md).

## Environment

The local development environment is:

```bash
cd /home/aditya/bude_vla
export MUJOCO_GL=egl
export PYTHONPATH=src
/home/aditya/venv-bude/bin/python -m unittest discover -s tests -v
```

Generated datasets, caches, checkpoints, logs, and videos remain local and are
ignored by Git.
