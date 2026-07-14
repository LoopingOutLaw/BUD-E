# BUD-E

BUD-E is a compact vision-language-action system for closed-loop pick-and-place
with the 6-DoF LeRobot SO-101 arm. It learns from dual-camera demonstrations in
MuJoCo, predicts task-space action chunks, and uses measured gripper feedback
for local recovery when a grasp fails.

**Project status: complete.** The retained v43 system exceeds the project's
80% acceptance target on 200 unseen random cube positions.

## Final Results

All results use the same seeded positions and the strict placement definition:
the cube must be released inside the bowl, below the rim, moving slowly, and
stable for eight consecutive policy steps.

| Evaluation | Success | Contact | Strict grasp |
| --- | ---: | ---: | ---: |
| VLA, one attempt | **162/200 (81.0%)** | 183/200 (91.5%) | 174/200 (87.0%) |
| VLA + local feedback recovery | **189/200 (94.5%)** | 199/200 (99.5%) | 196/200 (98.0%) |

On the paired benchmark, local recovery converted 27 previous failures into
successes with zero regressions. Twenty-six of those episodes invoked the
closed-loop grasp recovery; one completed because of the bounded longer
horizon. The final fixed-position video scored 7/8.

## Videos

These short checkpoint evaluations show the development path and final system.
Historical videos are qualitative milestones; the table above is the canonical
quantitative result.

| Stage | Video | What changed |
| --- | --- | --- |
| Camera-fixed baseline | [MP4](media/01_camera_fixed_baseline.mp4) | Correct camera, timing, and observation contracts |
| First complete learned picks | [MP4](media/02_first_complete_pick.mp4) | Spatially responsive shoulder control and full transport |
| Final v43 one-shot VLA | [MP4](media/03_v43_one_shot.mp4) | Fresh strict data, absolute TCP chunks, corrected placement metric |
| Final feedback recovery | [MP4](media/04_v43_feedback_recovery.mp4) | In-place reopen, realign, regrasp, verify, and continue |

## System Contract

The learned policy receives only signals available on a physical arm:

- current and previous 224x224 top-camera and wrist-camera RGB frames;
- five measured arm-joint positions and one gripper position;
- the language instruction;
- a three-value red-component observation derived from camera pixels.

It never receives MuJoCo cube coordinates, target-relative object vectors,
episode progress, simulator contacts, or success labels at inference.
Simulator state is restricted to scripted teachers, calibration fixtures,
diagnostics, and post-rollout metrics.

The output is a 16-step chunk of four absolute actions:

```text
[tcp_x, tcp_y, tcp_z, gripper]
```

An orientation-constrained damped least-squares IK controller converts each TCP
target into SO-101 joint commands. Demonstrations and deployment both run at
31.25 Hz, with four MuJoCo substeps per retained action.

## Model

The selected checkpoint contains 40,716,382 parameters:

- DINOv2 ViT-S/14 vision tower with a 12-channel dual-camera/history adapter;
- the last four DINOv2 transformer blocks fine-tuned;
- a compact BPE text transformer and task-domain soft prompts;
- affine proprioceptive and RGB-geometry projectors;
- an 8-layer, 256-wide multimodal policy transformer;
- a context action decoder producing 16 absolute TCP/gripper targets;
- a zero-initialized raw-geometry residual for precise spatial response.

The retained model is
`checkpoints/pick_v43_strict_geometry/pick_v43_strict_geometry_best.pt`,
selected at raw step 80,000 by strict closed-loop evaluation. Checkpoints,
datasets, frame caches, and logs are intentionally not stored in Git.

## Local Grasp Recovery

Recovery wraps the VLA only after a failed close. Successful VLA trajectories
pass through bit-for-bit unchanged.

1. The VLA requests closure.
2. The controller waits two policy frames and reads measured jaw position.
3. A blocked aperture verifies the grasp and returns control to the VLA.
4. An empty close reopens at the current TCP and backs away 55 mm.
5. Calibrated top-camera RGB reacquires the displaced cube.
6. RGB plus IK performs a local approach, descent, slow close, and bounded
   tighten.
7. Motor obstruction verifies the grasp before a local lift and fresh VLA
   replan.
8. Loss of aperture retries locally; exhausted retries abort rather than carry
   an empty gripper.

This is not a whole-task retry: the arm is not homed and the cube is not reset.
Runtime recovery does not read simulator object position or contact state. The
MuJoCo jaw threshold is not a real-hardware constant and must be calibrated on
the physical gripper.

## Installation

Python 3.11 or newer and a CUDA-capable PyTorch installation are recommended.
The final model was developed on an RTX 4060 Laptop GPU with 8 GiB VRAM.

```bash
git clone https://github.com/LoopingOutLaw/BUD-E.git
cd BUD-E
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[sim,dev]"
```

DINOv2 weights are downloaded by `timm` on first use. For headless MuJoCo:

```bash
export MUJOCO_GL=egl
export PYTHONPATH=src
```

## Reproduce the Final Pipeline

The final pipeline performs expert validation, fresh data recording,
policy-rate replay validation, task-space conversion, bounded cache creation,
training, checkpoint selection, broad benchmarking, and final video export.
It has no wall-clock timeout and is resumable at completed stages.

```bash
cd BUD-E
bash scripts/run_v43_strict_geometry.sh
```

The current recipe initializes from the retained v42 geometry checkpoint. Set
`INIT_CKPT` to an equivalent compatible checkpoint when reproducing on a
different machine.

Run the canonical local-recovery benchmark independently:

```bash
MUJOCO_GL=egl PYTHONPATH=src python scripts/benchmark_random_pick.py \\
  --ckpt checkpoints/pick_v43_strict_geometry/pick_v43_strict_geometry_best.pt \\
  --raw-weights --num-episodes 200 --max-steps 650 --max-tries 1 \\
  --local-grasp-retry --local-grasp-retries 2 --seed 4311 \\
  --min-success-rate 0.80
```

Generate the final fixed-position video:

```bash
mkdir -p demos/videos
MUJOCO_GL=egl PYTHONPATH=src python scripts/eval_pick_ball.py \\
  --ckpt checkpoints/pick_v43_strict_geometry/pick_v43_strict_geometry_best.pt \\
  --raw-weights --num-episodes 8 --max-steps 650 --max-tries 1 \\
  --local-grasp-retry --local-grasp-retries 2 \\
  --cube-positions '0.23,-0.02;0.25,0.00;0.27,0.02;0.29,0.04;0.31,-0.01;0.33,0.05;0.22,0.06;0.34,0.03' \\
  --out demos/videos/eval_pick_v43_local_retry.mp4
```

## Tests

```bash
MUJOCO_GL=egl PYTHONPATH=src python -m unittest discover -s tests -v
```

The suite covers observation compatibility, action conversion, cache sampling,
placement semantics, local retry transitions, RGB reacquisition, and the rule
that verified VLA actions remain unchanged.

## Repository Layout

```text
media/                                  Curated milestone and final videos
docs/pick_vla_training_notes.md         Final technical report
scripts/run_v43_strict_geometry.sh      End-to-end final pipeline
scripts/record_pick_episodes.py         Fresh demonstration recorder
scripts/validate_dataset_replay.py      Persisted-action replay gate
scripts/build_frame_cache.py            Bounded history-aware frame cache
scripts/train.py                        Training and strict checkpoint selection
scripts/benchmark_random_pick.py        Broad random-position benchmark
scripts/eval_pick_ball.py               Video evaluation
src/bude_vla/models/                    VLA model components
src/bude_vla/grasp_retry.py             Feedback-gated local recovery
src/bude_vla/visual_servo.py            RGB localization and homography
src/bude_vla/envs/                      SO-101 MuJoCo environment
tests/                                  Regression tests
```

## Technical Report

The final architecture, data contract, training recipe, failure analysis,
evaluation protocol, and sim-to-real requirements are documented in
[docs/pick_vla_training_notes.md](docs/pick_vla_training_notes.md).

## References

- [LeRobot SO-101](https://huggingface.co/docs/lerobot/main/en/so101)
- [LeRobot ACT](https://huggingface.co/docs/lerobot/act)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [OpenVLA](https://github.com/openvla/openvla)
- [OpenVLA-OFT](https://openvla-oft.github.io/)
- [ALOHA and ACT](https://tonyzhaozh.github.io/aloha/)

## License

See [LICENSE](LICENSE).
