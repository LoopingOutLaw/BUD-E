# BUD-E

BUD-E is a compact vision-language-action stack for an SO-101 / LeRobot-style
arm in MuJoCo. The active task is to pick a red cube and place it in the blue
target zone from top and wrist RGB, robot joint state, and a language command.

The learned policy is never given simulator cube coordinates. Its deployable
observation contract is camera pixels plus the six measured arm/gripper joint
positions. Simulator object state is used only by demonstration teachers and
post-rollout metrics.

## Current Status

V42 is the retained learned-policy baseline. It preserves camera and joint
geometry, predicts absolute TCP targets plus gripper control, and executes them
through IK. Its selected raw step-155k artifact is stored as
`checkpoints/pick_v42_affine_geometry/pick_v42_affine_geometry_best.pt`.

An outcome-audit found that all earlier evaluators declared success when a
previously grasped cube came within 50 mm of the bowl in XY. They did not
require release, low height, or settling, so videos stopped while the cube was
still held above the bowl. Historical v37-v42 success percentages are therefore
transport-arrival metrics and must not be compared to the corrected metric.

Success requires the cube center within 20 mm of the bowl center, below
50 mm, moving slowly, free of both finger pads, for eight consecutive policy
steps. V42 scores 4/8 on the fixed set with one attempt and 5/8 with one
same-scene retry. Its independent 200-position one-try baseline is 81/200
(40.5%) strict success, 132/200 contact, and 108/200 strict grasp. This broad
result, rather than the fixed eight positions, is the starting point for v43.

The corrected failures are spatially systematic. One-try strict success is
70.9% in central X but 11.8% at far X; the four Y bins score 24.4%, 41.7%,
60.0%, and 29.8%. Full-chunk diagnostics explain the gap: the expert endpoint
changes 1.021 m/m in X and 1.004 m/m in Y, while v42 changes only 0.799 and
0.401. Its endpoint p95 error is 47.7 mm, well outside grasp tolerance.

Inspection also found a demonstration bug. During CLOSE, the scripted teacher
recomputed its target from the live cube every step. When the moving jaw nudged
the cube, the whole arm chased it sideways instead of holding position and
closing against the static finger. Those trajectories taught the visible sweep
after contact. CLOSE now captures one fixed world-space anchor; the corrected
expert passed 100/100 strict randomized episodes, and its policy-rate recorder
passed 19/20 in the preflight pilot.

The active **pick_v43_strict_geometry** pipeline therefore regenerates all
demonstrations instead of extending flawed data. It keeps the VLA, DINOv2,
dual cameras, language, absolute TCP action chunks, and IK execution, and adds:

- a zero-initialized residual from measured joint state plus an RGB-derived red
  component centroid directly to action logits;
- exact reset and early-control cache anchors for every demonstrated cube;
- L1 action regression with increasing weight across the 16-action chunk;
- a fresh context action decoder with a separate learning rate;
- raw-weight strict grid checkpoint selection and bounded checkpoint retention.

The centroid is calculated from camera pixels and can run on the real camera;
it is not a simulator coordinate. Simulator cube state remains restricted to
the teacher and metrics. No progress clock, target coordinate, or simulator
contact is supplied to the policy.

## Run V43

    cd /home/aditya/bude_vla
    bash scripts/run_v43_strict_geometry.sh

The script has no timeout and is resumable at completed data/cache stages and
recent training checkpoints. It records up to 5,000 fresh attempts, rejects
failed policy-rate replays, validates both joint and converted `ee_abs`
controls, builds a 52k-frame cache, and trains for 220k steps. It then compares
five execution modes, logs separate 200-position one-try and two-try results,
and claims the target only if autonomous same-scene retries reach at least 80%
under the strict released-and-settled metric.

Correct fixed-set evaluation with one autonomous same-scene retry:

    cd /home/aditya/bude_vla
    MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python \
      scripts/eval_pick_ball.py \
      --ckpt checkpoints/pick_v43_strict_geometry/pick_v43_strict_geometry_best.pt \
      --raw-weights --num-episodes 8 --max-steps 450 --max-tries 2 \
      --cube-positions '0.23,-0.02;0.25,0.00;0.27,0.02;0.29,0.04;0.31,-0.01;0.33,0.05;0.22,0.06;0.34,0.03' \
      --out demos/videos/eval_pick_v43_strict_geometry.mp4

On a retry, only the arm is homed. The cube remains where the failed attempt
left it, policy history is cleared, and the policy reobserves the scene from
RGB. Simulator cube coordinates are not added to the policy input.

Watch progress:

    cd /home/aditya/bude_vla
    tail -F \
      logs/pick_v43_train.log \
      logs/pick_v43_chunk_geometry.log \
      logs/pick_v43_random_two_try.log

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

The v43 cache is bounded at approximately 31 GiB:

```text
52000 frames * 224 * 224 * (2 histories * 6 RGB channels) * 1 byte
```

Training uses batch size 4, gradient accumulation 8, and in-process data
loading with zero workers. The effective batch is 32 while the cache remains memory-mapped instead of
being copied into RAM. The runner checks free disk and available RAM before
expensive stages, sets a repository-local temporary directory, and does not
retain eval frames unless video recording is explicitly enabled. Training keeps
only three recent step checkpoints plus `best.pt` and `final.pt`.

## Repository Map

```text
scripts/run_v43_strict_geometry.sh      Active fresh-data strict VLA pipeline
scripts/run_v42_affine_geometry.sh      Archived affine-geometry baseline recipe
scripts/run_v37_camera_fixed.sh         Archived fresh-data baseline recipe
scripts/benchmark_scripted_pick.py      Strict expert/mechanics gate
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
