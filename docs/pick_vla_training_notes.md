# BUD-E Final Technical Report

This document describes the completed v43 SO-101 pick-and-place system. It
replaces the former version-by-version experiment diary. Historical percentages
that used the old transport-arrival metric are intentionally omitted because
they are not comparable to the final release-and-settle definition.

## 1. Objective And Acceptance

The task is:

> Pick up the red cube and place it in the blue target zone.

The learned system must infer object location from RGB, control the SO-101 from
measured robot state, and complete the task over random reachable cube
positions. Cube pose is never supplied to the learned policy or runtime
recovery controller.

The project acceptance gate is at least 80% strict success over 200 fresh random
positions. V43 reached 81.0% as a one-shot VLA and 94.5% with feedback-gated
local grasp recovery.

## 2. Final Quantitative Result

Checkpoint:

`checkpoints/pick_v43_strict_geometry/pick_v43_strict_geometry_best.pt`

The strict grid evaluator selected raw training step 80,000. The independent
paired benchmark used seed 4311.

| Metric | One-shot VLA | With local recovery |
| --- | ---: | ---: |
| Strict placement success | 162/200 (81.0%) | 189/200 (94.5%) |
| Any cube contact | 183/200 (91.5%) | 199/200 (99.5%) |
| Strict grasp | 174/200 (87.0%) | 196/200 (98.0%) |

Of 38 one-shot failures, 27 completed in the feedback run. Twenty-six invoked a
true local grasp retry and one needed only the bounded 650-step horizon. No
one-shot success regressed. Eleven episodes remained failures.

The eight-position diagnostic video scored 7/8. Five successes remained pure
VLA rollouts and two invoked local recovery.

## 3. Evaluation Semantics

Earlier evaluators stopped when a previously grasped cube moved near the bowl
in XY. That could mark success while the cube was still held above the target.
The final evaluator requires all of the following for eight consecutive policy
steps:

- cube center within 20 mm of the bowl center in XY;
- cube below 50 mm world height;
- cube translational speed below the settle threshold;
- no contact with either gripper pad;
- a grasp must have occurred earlier in the episode.

These semantics are shared by training-time evaluation,
`benchmark_random_pick.py`, and `eval_pick_ball.py`.

## 4. Runtime Observation And Action Contract

### Observations

| Signal | Shape | Source |
| --- | --- | --- |
| Top RGB | 2 x 3 x 224 x 224 | current and previous frame |
| Wrist RGB | 2 x 3 x 224 x 224 | current and previous frame |
| Proprioception | 6 | five arm joints plus gripper position |
| Instruction | up to 64 tokens | compact BPE tokenizer |
| RGB geometry | 3 | normalized red centroid and visibility |

The model does not receive:

- simulator cube XYZ;
- target-relative cube vectors;
- simulator contact or strict-grasp bits;
- episode phase or elapsed-progress features;
- scripted-controller state;
- post-rollout success labels.

RGB geometry is calculated from camera pixels. The top-camera pixel-to-workspace
homography is calibrated from known marker placements and does not query object
state during rollout.

### Actions

The policy predicts 16 future four-dimensional actions:

~~~text
[tcp_x, tcp_y, tcp_z, gripper]
~~~

The action space is `ee_abs`: absolute TCP position plus gripper target. An
orientation-constrained damped least-squares IK solver maps each target to the
five arm joints. Training demonstrations and policy execution both operate at
31.25 Hz with four MuJoCo substeps per retained action.

The final deployment mode executes the complete 16-action chunk before
replanning. Shorter horizons and temporal ensembling were benchmarked rather
than assumed; both reduced success for the selected checkpoint.

## 5. Model Architecture

The retained checkpoint contains 40,716,382 parameters.

| Component | Final configuration |
| --- | --- |
| Vision | pretrained DINOv2 ViT-S/14 |
| Vision input | 12 channels: two times x two RGB cameras |
| Vision adaptation | pretrained kernel on current top RGB; other channels zero-initialized |
| Vision fine-tuning | final four transformer blocks, norm, positions, adapter |
| Text | compact learned BPE transformer |
| Robot state | six-dimensional affine feature projector |
| Domain adaptation | 32 learned soft prompts |
| Fusion backbone | 8 layers, width 256, 8 attention heads, FFN 1024 |
| Action decoder | context transformer, 16 x 4 outputs |
| Spatial residual | zero-initialized joint-state plus RGB-geometry residual |
| Action normalization | persisted per-dimension bounds from recorded data |

The repository retains the flow-matching head for architecture compatibility,
but v43 trains the deterministic context action decoder with
`flow_loss_weight=0`. This was the more stable representation for precise
absolute task-space imitation on the available data and hardware.

## 6. Data Pipeline

V43 regenerated demonstrations instead of extending earlier datasets.

1. The scripted expert must pass at least 98/100 randomized strict episodes.
2. Up to 5,000 fresh successful demonstrations are recorded.
3. The controller plans at 125 Hz and retains every fourth action.
4. Each retained sequence is replayed at the exact 31.25 Hz policy rate.
5. Failed policy-rate replays are rejected before writing.
6. Joint targets are converted to absolute TCP targets.
7. Converted actions must pass at least 95% persisted-action replay.
8. A 52,000-frame memory-mapped cache is built with exact reset anchors and
   balanced early, approach, and late-task coverage.

Only successful trajectories enter the final base dataset. Mild XY/Z recovery,
light nudge recovery, and a small retry fraction provide corrective behavior
without allowing retries to dominate the main strategy.

### Corrected demonstration defects

The following defects were fixed before the final recording:

- camera framing did not cover the complete training workspace;
- red fingertip debug geometry confused red-object localization;
- demonstration and deployment action rates did not match;
- the pretrained DINO adapter was attached to the oldest rather than current
  top frame;
- RGB inputs lacked the expected pretrained normalization;
- progress proprio leaked scripted episode time into inference;
- stale contact could count as a grasp;
- the teacher chased a cube laterally while closing instead of holding a fixed
  final grasp anchor;
- some recovery paths lifted toward stale pre-retry positions;
- evaluation retained unnecessary frames and exhausted system RAM.

Training loss had previously fallen despite zero closed-loop success because
these were data and execution contract failures, not optimization failures.

## 7. Final Training Recipe

| Setting | Value |
| --- | ---: |
| Image size | 224 |
| History frames | 2 |
| Chunk size | 16 |
| Batch size | 4 |
| Gradient accumulation | 8 |
| Effective batch | 32 |
| Data workers | 0 |
| Planned steps | 220,000 |
| Main learning rate | 2e-5 |
| Action decoder learning rate | 1e-4 |
| DINOv2 learning rate | 1e-7 |
| BC objective | masked L1 |
| BC weight | 8.0 |
| Chunk-end weight | 4.0 |
| Gripper weight | 5.0 |
| Flow weight | 0.0 |
| EMA | disabled |
| Evaluation | 64 episodes on an 8 x 8 grid every 10k steps |

Checkpoint selection uses strict closed-loop success, not final supervised
loss. Training keeps three recent snapshots plus `best.pt` and `final.pt`.
This matters because later checkpoints can overfit even while training loss
continues to decrease.

## 8. Feedback-Gated Local Grasp Recovery

The user-facing retry requirement was: if grasping fails, do not continue to
transport; reopen and retry from the current scene. The old implementation
homed the arm and replayed the entire task, which recovered only 8 of 38
failures.

The final state machine in `src/bude_vla/grasp_retry.py` implements the
requested local behavior:

1. **Policy:** pass VLA commands through unchanged.
2. **Settle:** after a close request, wait two measured policy frames.
3. **Verify:** treat a blocked jaw aperture as a grasp.
4. **Open:** if empty, reopen without moving to home.
5. **Back off:** lift 55 mm at the current TCP and clear stale action chunks.
6. **Reacquire:** locate the displaced red cube from top-camera RGB.
7. **Align:** use RGB plus IK for a bounded local approach and descent.
8. **Probe:** close slowly while monitoring measured motor obstruction.
9. **Tighten:** apply only a bounded additional close after obstruction.
10. **Lift:** move to the demonstrated post-grasp height.
11. **Replan:** return control to a fresh VLA chunk.
12. **Abort:** after two failed local cycles, stop instead of carrying empty.

Successful VLA trajectories remain bit-for-bit unchanged. The controller never
homes the arm, resets the cube, or receives MuJoCo cube/contact state.

### Grasp verification calibration

In simulation, empty hard closure settled near -0.175 rad. Sixty successful
cube grasps first registered between 0.1635 and 0.1968 rad, with median 0.1803.
The simulated default threshold of 0.08 lies between those distributions.

That number must not be copied directly to a physical SO-101. Real deployment
must measure empty-close and object-blocked distributions for the assembled
gripper, then choose a threshold with margin. Servo position, load/current if
available, timeout, and a conservative lift check should all feed the hardware
safety implementation.

## 9. Rejected Approaches

The following were measured and intentionally not retained:

- more steps on flawed data;
- hidden cube position in model proprioception;
- episode-progress proprioception;
- whole-task homing retries;
- deterministic VLA replan after the same local miss;
- blind radial search around an estimated cube;
- forced lift/transport without verified grasp;
- shortening execution horizon for a model trained on full chunks;
- temporal ensembling without paired validation;
- DAgger datasets collected from policies with incorrect camera/time contracts;
- pure end-effector delta control, which lost broad reaching accuracy.

These experiments remain represented in Git history where useful, but their
version-specific launch scripts, logs, checkpoints, caches, and videos are not
part of the completed source snapshot.

## 10. Reproduction

Run the complete no-time-limit pipeline:

~~~bash
cd BUD-E
bash scripts/run_v43_strict_geometry.sh
~~~

The pipeline checks free disk and available RAM, refuses concurrent data or
training processes, resumes completed data/cache stages, and resumes the newest
compatible checkpoint after interruption.

Run tests:

~~~bash
MUJOCO_GL=egl PYTHONPATH=src python -m unittest discover -s tests -v
~~~

Run the final benchmark:

~~~bash
MUJOCO_GL=egl PYTHONPATH=src python scripts/benchmark_random_pick.py --ckpt checkpoints/pick_v43_strict_geometry/pick_v43_strict_geometry_best.pt --raw-weights --num-episodes 200 --max-steps 650 --max-tries 1 --local-grasp-retry --local-grasp-retries 2 --seed 4311 --min-success-rate 0.80
~~~

## 11. Curated Videos

| File | Purpose |
| --- | --- |
| `media/01_camera_fixed_baseline.mp4` | early corrected training baseline |
| `media/02_first_complete_pick.mp4` | first complete learned pick-and-place milestone |
| `media/03_v43_one_shot.mp4` | final one-shot learned policy |
| `media/04_v43_feedback_recovery.mp4` | final in-place closed-loop grasp recovery |

Generated evaluation videos belong under `demos/videos/` and remain ignored.
Only this four-video set is tracked in Git.

## 12. Artifact And Storage Policy

Tracked:

- source, tests, final runner, technical documentation, URDF/assets;
- four compressed representative MP4 videos.

Local and ignored:

- `data/`: recorded episodes and frame caches;
- `checkpoints/`: model snapshots;
- `logs/`: training and evaluation logs;
- `demos/`: generated videos;
- temporary files and Python caches.

The final cache is approximately 31 GiB and is memory-mapped. Training uses
zero data workers to avoid duplicating the cache in RAM on the development
laptop.

## 13. Physical SO-101 Deployment

The simulation policy is not a drop-in command for uncalibrated hardware.
Before physical execution:

1. calibrate all joint directions, offsets, and hard limits;
2. calibrate top and wrist cameras and refit the top-camera homography;
3. calibrate empty and blocked gripper feedback thresholds;
4. limit TCP velocity, joint velocity, gripper force, and recovery attempts;
5. add an operator emergency stop and workspace exclusion zones;
6. validate perception and IK with the arm disabled;
7. replay slowly above the table before enabling contact;
8. collect real demonstrations and fine-tune for the visual/physics domain gap.

The deployable observation design is compatible with this transition: cameras,
joint encoders, language, and servo feedback are real signals. MuJoCo-only
metrics remain outside control.

## References

- [LeRobot SO-101 documentation](https://huggingface.co/docs/lerobot/main/en/so101)
- [LeRobot hardware integration](https://huggingface.co/docs/lerobot/main/en/integrate_hardware)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [OpenVLA](https://github.com/openvla/openvla)
- [OpenVLA-OFT](https://openvla-oft.github.io/)
- [ALOHA and ACT](https://tonyzhaozh.github.io/aloha/)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
