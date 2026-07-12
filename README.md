# BUD-E

BUD-E is a compact vision-language-action stack for an SO-101 / LeRobot-style
arm in MuJoCo. The active task is to pick a red cube and place it in the blue
target zone from top and wrist RGB, robot joint state, and a language command.

The learned policy is never given simulator cube coordinates. Its deployable
observation contract is camera pixels plus the six measured arm/gripper joint
positions. Simulator object state is used only by demonstration teachers and
post-rollout metrics.

## Current Status

The corrected v37 run is complete. Its original pipeline summary of 0% was an
evaluation error, not the true final-policy result:

- `pick_v37_camera_fixed_best.pt` was the 5,000-step checkpoint because all
  ensemble-mode evals tied at zero and ties never advanced the best file.
- Evaluation loaded EMA weights even though the final raw weights were better.
- Temporal ensembling or replanning before all 16 trained actions were executed
  sharply reduced contact.

With the final 25,000-step raw weights and native 16-action chunk execution, a
fresh 50-position random benchmark produced 5/50 successful grasp-and-place
rollouts, 23/50 contact episodes, and 7/50 strict grasps. This is the first
controlled proof that the corrected camera/action pipeline learned the task,
but 10% success is not the target.

The remaining data bottleneck is also measured: the 6,000-row v37 cache covered
only 2,986 of 3,784 demonstrations. Exactly 798 randomized cube placements had
no training observation in the cache. The active experiment is therefore
`pick_v38_broad_cache`, which continues from the successful raw v37 weights on
a 64,000-row cache with guaranteed coverage of every demonstration.

The learned policy is still never given simulator cube coordinates, contact,
grasp state, or an episode clock. Its runtime inputs remain RGB, joint encoders,
and language.

## Run V38

Run the guarded continuation from the repository root:

```bash
cd /home/aditya/bude_vla
bash scripts/run_v38_broad_cache.sh
```

There is no timeout. The runner:

1. Rechecks the camera-only expert and persisted-action replay gates.
2. Reuses the verified 3,784-episode v37 dataset.
3. Builds 64,000 history-stacked rows with at least 12 phase-stratified rows
   from every episode.
4. Initializes from the v37 raw `model_state_dict`, never its lagging EMA.
5. Trains for 100,000 microsteps with a lower continuation learning rate.
6. Runs 30 native-chunk closed-loop episodes every 10,000 steps and preserves
   the best checkpoint, including later checkpoints when scores tie.
7. Runs a 150-position random benchmark and an 8-position video evaluation.

Watch it from another terminal:

```bash
cd /home/aditya/bude_vla
tail -F \
  logs/pick_v38_visual_expert_bench.log \
  logs/pick_v38_replay.log \
  logs/pick_v38_cache.log \
  logs/pick_v38_train.log \
  logs/pick_v38_random_bench.log
```

The selected checkpoint is written to
`logs/pick_v38_selected_checkpoint.txt`; the video is
`demos/videos/eval_pick_v38_broad_cache.mp4`.

## Corrected V37 Result

Paired diagnostics on seed 3710 established the inference contract:

| Policy / execution | Success | Contact | Strict grasp |
| --- | ---: | ---: | ---: |
| v37 raw 25k, native chunk, 50 positions | 5/50 | 23/50 | 7/50 |
| v37 raw 20k, native chunk, first 30 positions | 0/30 | 13/30 | 2/30 |
| v37 raw 25k, horizon 8, first 20 positions | 0/20 | 2/20 | 1/20 |
| v37 raw 25k, contact reflex, first 30 positions | 4/30 | 15/30 | 8/30 |

The 20k-to-25k improvement shows optimization was still helping. The horizon-8
collapse shows this model must execute the full chunk it was trained to emit.
The simulator-only contact reflex improved grasp count but not completed place
success, so it is diagnostic only and is not part of the deployable policy.

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
- Native rollout executes all sixteen predicted actions before replanning.
  Temporal ensembling is disabled because paired benchmarks showed it was
  harmful for this checkpoint.

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
The use of action chunks is aligned with
[LeRobot's ACT guidance](https://huggingface.co/docs/lerobot/act), while the
repeated random-position coverage follows
[LeRobot's SmolVLA data guidance](https://huggingface.co/docs/lerobot/smolvla).

## Memory And Storage

The v38 cache is approximately 36 GiB:

```text
64000 frames * 224 * 224 * (2 histories * 6 RGB channels) * 1 byte
```

Training uses batch size 4, gradient accumulation 8, and one worker. The
effective batch is 32 while the 36 GiB cache remains memory-mapped instead of
being copied into RAM. The runner checks free disk and available RAM before
expensive stages, sets a repository-local temporary directory, and does not
retain eval frames unless video recording is explicitly enabled.

## Repository Map

```text
scripts/run_v38_broad_cache.sh        Active broad-cache raw-weight continuation
scripts/run_v37_camera_fixed.sh       Reproducible fresh v37 baseline pipeline
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
