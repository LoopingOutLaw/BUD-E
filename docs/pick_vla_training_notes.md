# BUD-E Final Technical Report

> **Complete documentation for the v43 SO-101 pick-and-place system.**
>
> This document replaces the former version-by-version experiment diary. Historical percentages that used the old transport-arrival metric are intentionally omitted because they are not comparable to the final release-and-settle definition.

---

## Table of Contents

1. [Objective & Acceptance Criteria](#1-objective--acceptance-criteria)
2. [Final Quantitative Results](#2-final-quantitative-results)
3. [Evaluation Semantics](#3-evaluation-semantics)
4. [Runtime Observation & Action Contract](#4-runtime-observation--action-contract)
5. [Model Architecture](#5-model-architecture)
6. [Data Pipeline](#6-data-pipeline)
7. [Training Recipe](#7-training-recipe)
8. [Feedback-Gated Local Grasp Recovery](#8-feedback-gated-local-grasp-recovery)
9. [Rejected Approaches](#9-rejected-approaches)
10. [Reproduction](#10-reproduction)
11. [Curated Videos](#11-curated-videos)
12. [Artifact & Storage Policy](#12-artifact--storage-policy)
13. [Physical SO-101 Deployment](#13-physical-so-101-deployment)

---

## 1. Objective & Acceptance Criteria

**Task:** Pick up the red cube and place it in the blue target zone.

The learned system must infer object location from RGB, control the SO-101 from measured robot state, and complete the task over random reachable cube positions. **Cube pose is never supplied to the learned policy or runtime recovery controller.**

| Gate | Threshold | v43 Result |
|------|-----------|------------|
| Strict success (one-shot) | ≥ 80% | **81.0%** ✅ |
| Strict success (with recovery) | — | **94.5%** |

---

## 2. Final Quantitative Results

**Checkpoint:** `checkpoints/pick_v43_strict_geometry/pick_v43_strict_geometry_best.pt`

The strict grid evaluator selected raw training step **80,000**. The independent paired benchmark used seed **4311**.

| Metric | One-shot VLA | With Local Recovery |
|--------|:------------:|:-------------------:|
| **Strict placement success** | **162/200 (81.0%)** | **189/200 (94.5%)** |
| Any cube contact | 183/200 (91.5%) | 199/200 (99.5%) |
| Strict grasp | 174/200 (87.0%) | 196/200 (98.0%) |

### Failure Analysis

- **38** one-shot failures → **27** completed with feedback (26 true local retries, 1 bounded horizon)
- **Zero** one-shot successes regressed
- **11** episodes remained failures after recovery
- Diagnostic video (8 fixed positions): **7/8** (5 pure VLA, 2 with recovery)

---

## 3. Evaluation Semantics

### Strict Placement Definition

Earlier evaluators stopped when a previously grasped cube moved near the bowl in XY. This could mark success while the cube was still held above the target. The final evaluator requires **all** of the following for **8 consecutive policy steps**:

| Criterion | Threshold |
|-----------|-----------|
| Cube center to bowl center (XY) | ≤ 20 mm |
| Cube height (world Z) | ≤ 50 mm |
| Translational speed | < settle threshold |
| No gripper contact | Both pads clear |
| Grasp occurred | Earlier in episode |

These semantics are shared by training-time evaluation, `benchmark_random_pick.py`, and `eval_pick_ball.py`.

---

## 4. Runtime Observation & Action Contract

### 4.1 Observations

| Signal | Shape | Source | Notes |
|--------|-------|--------|-------|
| Top RGB | 2 × 3 × 224 × 224 | Current + previous frame | History-stacked |
| Wrist RGB | 2 × 3 × 224 × 224 | Current + previous frame | History-stacked |
| Proprioception | 6 | Measured encoders | 5 arm joints + gripper |
| Instruction | ≤ 64 tokens | Compact BPE tokenizer | Language task spec |
| RGB geometry | 3 | Camera pixels | Normalized red centroid + visibility |

**The model does NOT receive:**
- ❌ Simulator cube XYZ
- ❌ Target-relative cube vectors
- ❌ Simulator contact or strict-grasp bits
- ❌ Episode phase or elapsed-progress features
- ❌ Scripted-controller state
- ❌ Post-rollout success labels

RGB geometry is calculated from camera pixels. The top-camera pixel-to-workspace homography is calibrated from known marker placements and does not query object state during rollout.

### 4.2 Actions

The policy predicts **16 future four-dimensional actions**:

```text
[tcp_x, tcp_y, tcp_z, gripper]
```

| Property | Value |
|----------|-------|
| Action space | `ee_abs` — absolute TCP position + gripper target |
| Controller | Orientation-constrained damped least-squares IK |
| Policy rate | 31.25 Hz |
| MuJoCo substeps | 4 per retained action |
| Execution mode | Full 16-action chunk before replanning |

> **Note:** Shorter horizons and temporal ensembling were benchmarked rather than assumed; both reduced success for the selected checkpoint.

---

## 5. Model Architecture

**Total parameters:** 40,716,382

| Component | Configuration |
|-----------|---------------|
| **Vision** | Pretrained DINOv2 ViT-S/14 |
| Vision input | 12 channels: 2 times × 2 RGB cameras |
| Vision adaptation | Pretrained kernel on current top RGB; other channels zero-initialized |
| Vision fine-tuning | Final 4 transformer blocks + norm + positions + adapter |
| **Text** | Compact learned BPE transformer |
| **Robot state** | 6-dimensional affine feature projector |
| **Domain adaptation** | 32 learned soft prompts |
| **Fusion backbone** | 8 layers, width 256, 8 attention heads, FFN 1024 |
| **Action decoder** | Context transformer, 16 × 4 outputs |
| **Spatial residual** | Zero-initialized joint-state + RGB-geometry residual |
| **Action normalization** | Persisted per-dimension bounds from recorded data |

### Design Notes

- The repository retains the flow-matching head for architecture compatibility, but v43 trains the deterministic context action decoder with `flow_loss_weight=0`. This was the more stable representation for precise absolute task-space imitation on the available data and hardware.
- The spatial residual provides direct kinematic-state conditioning at the action decoder, avoiding the need for the transformer to preserve exact joint values through all attention layers.

---

## 6. Data Pipeline

V43 regenerated demonstrations instead of extending earlier datasets. Each stage gates the next.

### 6.1 Pipeline Stages

```
Expert Gate (≥98/100)
    ↓
Fresh Recording (≤5,000 episodes)
    ↓
Policy-Rate Replay Validation (reject failures)
    ↓
Task-Space Conversion (joint → ee_abs)
    ↓
Replay Gate (≥95% success)
    ↓
Frame Cache (52,000 frames, balanced coverage)
    ↓
Training
```

### 6.2 Stage Details

| Stage | Requirement |
|-------|-------------|
| Expert gate | ≥ 98/100 randomized strict episodes |
| Recording | Up to 5,000 fresh successful demonstrations |
| Control rate | 125 Hz planner, every 4th action retained |
| Replay validation | Failed policy-rate replays rejected before writing |
| Conversion | Joint targets → absolute TCP targets |
| Conversion gate | ≥ 95% persisted-action replay success |
| Frame cache | 52,000 frames, exact reset anchors, balanced early/approach/late coverage |

### 6.3 Corrected Demonstration Defects

The following defects were identified and fixed before final recording:

| # | Defect | Impact |
|---|--------|--------|
| 1 | Camera framing incomplete | Lost workspace coverage |
| 2 | Red fingertip debug geometry | Confused red-object localization |
| 3 | Action rate mismatch between demo and deploy | Execution contract violation |
| 4 | DINO adapter on oldest (not current) top frame | Stale visual features |
| 5 | Missing ImageNet normalization | Distribution shift |
| 6 | Progress proprio leaked scripted time | Privileged information at inference |
| 7 | Stale contact counted as grasp | False positive grasp detection |
| 8 | Teacher chased cube while closing | Unstable final grasp |
| 9 | Recovery lifted toward stale positions | Incorrect retry trajectories |
| 10 | Evaluation retained unnecessary frames | RAM exhaustion |

> **Critical insight:** Training loss had previously fallen despite zero closed-loop success because these were **data and execution contract failures**, not optimization failures.

---

## 7. Training Recipe

| Setting | Value |
|---------|------:|
| Image size | 224 |
| History frames | 2 |
| Chunk size | 16 |
| Batch size | 4 |
| Gradient accumulation | 8 |
| **Effective batch** | **32** |
| Data workers | 0 |
| Planned steps | 220,000 |
| Main learning rate | 2e-5 |
| Action decoder learning rate | 1e-4 |
| DINOv2 learning rate | 1e-7 |
| BC objective | Masked L1 |
| BC weight | 8.0 |
| Chunk-end weight | 4.0 |
| Gripper weight | 5.0 |
| Flow weight | 0.0 |
| EMA | Disabled |
| Evaluation | 64 episodes on 8×8 grid every 10k steps |

### Checkpoint Selection

- Uses **strict closed-loop success**, not final supervised loss
- Keeps 3 recent snapshots + `best.pt` + `final.pt`
- Later checkpoints can overfit even while training loss continues to decrease

---

## 8. Feedback-Gated Local Grasp Recovery

### 8.1 Design Requirement

> "If grasping fails, do not continue to transport; reopen and retry from the current scene."

The old implementation homed the arm and replayed the entire task — recovering only 8 of 38 failures. The final state machine implements **true local behavior**.

### 8.2 State Machine

| Step | Phase | Action |
|------|-------|--------|
| 1 | **Policy** | Pass VLA commands through unchanged |
| 2 | **Settle** | After close request, wait 2 measured policy frames |
| 3 | **Verify** | Blocked jaw aperture → grasp; empty → recovery |
| 4 | **Open** | If empty, reopen without moving to home |
| 5 | **Back off** | Lift 55 mm at current TCP; clear stale action chunks |
| 6 | **Reacquire** | Locate displaced red cube from top-camera RGB |
| 7 | **Align** | RGB + IK for bounded local approach and descent |
| 8 | **Probe** | Close slowly while monitoring measured motor obstruction |
| 9 | **Tighten** | Bounded additional close after obstruction |
| 10 | **Lift** | Move to demonstrated post-grasp height |
| 11 | **Replan** | Return control to fresh VLA chunk |
| 12 | **Abort** | After 2 failed local cycles, stop (don't carry empty) |

### 8.3 Key Properties

- ✅ Successful VLA trajectories remain **bit-for-bit unchanged**
- ✅ Controller **never homes** the arm
- ✅ Controller **never resets** the cube
- ✅ Controller **never receives** MuJoCo cube/contact state

### 8.4 Grasp Verification Calibration

| Condition | Measured Position |
|-----------|-------------------|
| Empty hard closure | ~ -0.175 rad |
| Successful cube grasps (60 samples) | 0.1635 – 0.1968 rad (median 0.1803) |
| **Default threshold** | **0.08 rad** |

> ⚠️ **This threshold must be recalibrated on physical hardware.** Real deployment must measure empty-close and object-blocked distributions for the assembled gripper, then choose a threshold with margin. Servo position, load/current, timeout, and conservative lift check should all feed the hardware safety implementation.

---

## 9. Rejected Approaches

The following were **measured and intentionally not retained**:

| Approach | Why Rejected |
|----------|--------------|
| More steps on flawed data | Fundamental contract issues |
| Hidden cube position in proprio | Privileged state, not deployable |
| Episode-progress proprio | Same issue |
| Whole-task homing retries | Only 8/38 recovery rate |
| Deterministic VLA replan after miss | No feedback, repeated failures |
| Blind radial search around estimate | Unreliable, no visual confirmation |
| Forced lift/transport without verified grasp | High drop rate |
| Shortening horizon for full-chunk model | Reduced success |
| Temporal ensembling without validation | Assumed, not measured |
| DAgger from policies with wrong contracts | Garbage in, garbage out |
| Pure end-effector delta control | Lost broad reaching accuracy |

These experiments remain in Git history where useful, but their version-specific scripts, logs, checkpoints, caches, and videos are not part of the completed source snapshot.

---

## 10. Reproduction

### Full Pipeline

```bash
cd BUD-E
bash scripts/run_v43_strict_geometry.sh
```

The pipeline:
- ✅ Checks free disk (≥70 GiB) and available RAM (≥6 GiB)
- ✅ Refuses concurrent data or training processes
- ✅ Resumes completed data/cache stages
- ✅ Resumes newest compatible checkpoint after interruption

### Independent Benchmark

```bash
MUJOCO_GL=egl PYTHONPATH=src python scripts/benchmark_random_pick.py \
  --ckpt checkpoints/pick_v43_strict_geometry/pick_v43_strict_geometry_best.pt \
  --raw-weights --num-episodes 200 --max-steps 650 --max-tries 1 \
  --local-grasp-retry --local-grasp-retries 2 --seed 4311 \
  --min-success-rate 0.80
```

### Tests

```bash
MUJOCO_GL=egl PYTHONPATH=src python -m unittest discover -s tests -v
```

---

## 11. Curated Videos

| File | Purpose |
|------|---------|
| `media/01_camera_fixed_baseline.mp4` | Early corrected training baseline |
| `media/02_first_complete_pick.mp4` | First complete learned pick-and-place milestone |
| `media/03_v43_one_shot.mp4` | Final one-shot learned policy |
| `media/04_v43_feedback_recovery.mp4` | Final in-place closed-loop grasp recovery |

Generated evaluation videos belong under `demos/videos/` and remain ignored. Only this four-video set is tracked in Git.

---

## 12. Artifact & Storage Policy

### Tracked in Git

- Source, tests, final runner, technical documentation
- URDF and assets
- Four compressed representative MP4 videos
- Architecture diagram (`media/bude_architecture.png`)

### Local / Ignored

| Path | Contents | Approx. Size |
|------|----------|-------------|
| `data/` | Recorded episodes and frame caches | ~31 GiB |
| `checkpoints/` | Model snapshots | Variable |
| `logs/` | Training and evaluation logs | Variable |
| `demos/` | Generated videos | Variable |

The final cache is memory-mapped. Training uses zero data workers to avoid duplicating the cache in RAM on the development laptop.

---

## 13. Physical SO-101 Deployment

> ⚠️ **The simulation policy is NOT a drop-in command for uncalibrated hardware.**

Before physical execution:

| Step | Action |
|------|--------|
| 1 | Calibrate all joint directions, offsets, and hard limits |
| 2 | Calibrate top and wrist cameras; refit top-camera homography |
| 3 | Calibrate empty and blocked gripper feedback thresholds |
| 4 | Limit TCP velocity, joint velocity, gripper force, and recovery attempts |
| 5 | Add operator emergency stop and workspace exclusion zones |
| 6 | Validate perception and IK with the arm disabled |
| 7 | Replay slowly above the table before enabling contact |
| 8 | Collect real demonstrations and fine-tune for the visual/physics domain gap |

The deployable observation design is compatible with this transition: cameras, joint encoders, language, and servo feedback are real signals. MuJoCo-only metrics remain outside control.

---

## References

- [LeRobot SO-101 documentation](https://huggingface.co/docs/lerobot/main/en/so101)
- [LeRobot hardware integration](https://huggingface.co/docs/lerobot/main/en/integrate_hardware)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [OpenVLA](https://github.com/openvla/openvla)
- [OpenVLA-OFT](https://openvla-oft.github.io/)
- [ALOHA and ACT](https://tonyzhaozh.github.io/aloha/)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
