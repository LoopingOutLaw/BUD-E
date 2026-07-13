# BUD-E

BUD-E is a compact vision-language-action stack for an SO-101 / LeRobot-style
arm in MuJoCo. The active task is to pick a red cube and place it in the blue
target zone from top and wrist RGB, robot joint state, and a language command.

The learned policy is never given simulator cube coordinates. Its deployable
observation contract is camera pixels plus the six measured arm/gripper joint
positions. Simulator object state is used only by demonstration teachers and
post-rollout metrics.

## Current Status

V41 is the strongest completed learned policy. Its selected step-105k absolute
task-space checkpoint scored **89/200 success (44.5%)**, 145/200 contact
(72.5%), and 120/200 strict grasp (60.0%) on random workspace positions. This
more than doubled v39's 20.7% joint-space result. The fixed set remained 3/8.

The remaining error is spatially systematic: success is 63.2% in the central-X
quarter but 14.7% in far X, and 73.3% in positive-central Y but 23.2% in
negative Y. This is not missing data: all 3,784 demonstrations are represented
in the 64k cache and both episode and cache coverage are uniform by workspace.

Inspection exposed an architectural information-loss bug. Compact vectors
were passed through LayerNorm before projection, including the RGB-derived
`[pixel_x, pixel_y, valid]` centroid. Per-sample LayerNorm removes cross-feature
mean and scale, so distinct 2D locations can become identical before the first
learned layer. The same problem affected the six joint-state values.

The active experiment is **pick_v42_affine_geometry**. It keeps the VLA,
DINOv2, dual cameras, language, absolute TCP action chunks, and IK execution,
but replaces those lossy transforms with learnable per-feature affine scaling.
It also gives the action decoder a direct deployable joint-state embedding.
No cube coordinates, progress clock, simulator contacts, or other privileged
state are added.

## Run V42

    cd /home/aditya/bude_vla
    bash scripts/run_v42_affine_geometry.sh

V42 reuses the v41 task-space dataset and existing 36 GiB image cache. It
retains the trained DINOv2/transformer trunk, deliberately rebuilds only the
proprio, pixel-centroid, and context-action modules, and selects checkpoints on
a deterministic 6x6 grid. After training it compares EMA/raw weights and four
receding-horizon modes on identical positions, runs a 200-position acceptance
benchmark, prints stage and workspace failure bins, and writes a video. It does
not claim success unless the independent random benchmark reaches 80%.

Watch progress:

    cd /home/aditya/bude_vla
    tail -F \
      logs/pick_v42_train.log \
      logs/pick_v42_task_space_sensitivity.log \
      logs/pick_v42_random_bench.log

## V39 Completed Result

The training evaluator selected step 35k at 9/40. A paired 100-position
selection diagnostic scored 25/100, ahead of step 45k at 22/100. The final
independent 150-position benchmark scored 31/150 success (20.7%), 79/150
contact (52.7%), and 48/150 strict grasp (32.0%). The fixed video scored 3/8
and showed complete visual approach, grasp, lift, rotation, transport, and
placement. Rotation toward the right after lift is expected because that is the
direction of the target zone.

Remaining success by workspace was 6% for negative Y versus 23-32% for positive
Y, and 11% in the far-X quarter versus 34% in the near-X quarter. Across X, the
expert shoulder-lift target spans 0.134 rad; v39 spans only 0.007 rad and uses
elbow motion as an incomplete substitute. Any next iteration should target this
radial shoulder-lift response rather than extend unchanged training.

## V38 Result

Training-time success on the fixed 30-position seed rose from 0/30 at 10k to
5/30 at both 70k and 90k, then fell to 3/30 at 100k. The runner correctly
selected the raw 90k checkpoint. Its independent 150-position result was:

```text
success:       21/150 (14.0%)
contact:       64/150 (42.7%)
strict grasp:  36/150 (24.0%)
```

The fixed video had strict grasps in two episodes but no completed placements,
which is why it printed 0/8 despite the broader benchmark succeeding.

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

Training uses batch size 4, gradient accumulation 8, and in-process data
loading with zero workers. The effective batch is 32 while the 36 GiB cache remains memory-mapped instead of
being copied into RAM. The runner checks free disk and available RAM before
expensive stages, sets a repository-local temporary directory, and does not
retain eval frames unless video recording is explicitly enabled.

## Repository Map

```text
scripts/run_v42_affine_geometry.sh     Active information-preserving VLA pipeline
scripts/run_v41_ee_abs.sh              Completed absolute task-space baseline
scripts/run_v40_radial_precision.sh Completed joint-space radial experiment
scripts/run_v39_shoulder_precision.sh Completed shoulder-pan precision run
scripts/run_v38_broad_cache.sh        Completed broad-cache baseline
scripts/run_v37_camera_fixed.sh       Reproducible fresh v37 baseline pipeline
scripts/benchmark_visual_servo_pick.py Camera-only perception/mechanics gate
scripts/record_pick_episodes.py        Fresh demonstration recorder
scripts/validate_dataset_replay.py     Persisted-action contract gate
scripts/build_frame_cache.py           Bounded frame/history cache builder
scripts/convert_dataset_to_ee_delta.py Task-space action relabeler
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
