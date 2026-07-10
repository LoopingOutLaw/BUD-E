# BUD-E

BUD-E is a compact Vision-Language-Action training stack for an SO-101 / LeRobot-style 6-DoF arm in MuJoCo. The active target is red-cube pick-and-place from dual-camera images, proprioception, contact state, and a language instruction.

The learned policy is intentionally not given privileged cube coordinates. Scripted demonstrators may use simulator state to generate successful examples, but the policy runs from visual observations and robot-side proprio/contact signals only.

## Current Status

The current experimental line is `pick_v35_ee_delta`, which pivots from joint-target actions to end-effector delta actions after v34 showed gripper timing was not the main blocker. Historical v33 intervention DAgger remains documented below.

Previously, the best joint-action line was `pick_v33_intervention_dagger`, trained from v31 with
scripted-intervention DAgger data. It is not solved yet, but it is the most
informative run so far: the policy reaches/touches more often, while strict
two-pad grasp remains the blocker.

What is working now:

- DINOv2 vision at 224 px.
- Dual-camera observations with 4-frame history.
- 10D proprioception without the old progress/time shortcut.
- Context/perception-conditioned action decoding.
- Mixed v26/v27/v28 training to preserve reaching while adding precision and recovery.
- Recovery-jitter, descent-depth, retry, and nudge/backoff demonstration support.
- Contact-focused cache sampling through `scripts/build_frame_cache.py --phase-ranges`.
- 10D contact-aware proprio with `any_pad_contact` before strict two-pad `is_grasping`.
- DAgger collection through `scripts/collect_dagger_pick.py`, where the learned policy visits states and an IK expert labels corrective actions.
- Random-position benchmarking through `scripts/benchmark_random_pick.py`, so progress is measured beyond the old 8 fixed eval positions.

Observed behavior after v33: the intervention dataset itself is strong
(1480 successful expert-intervention episodes with about 54% any-contact frames
and 51% strict-grasp frames), and random rollout contact improved to 45/150
episodes. However, autonomous strict grasp remains 0/150 and fixed 8-position
eval remains 0/8. The current bottleneck is no longer lack of grasp examples in
the dataset; it is converting visual/contact approach into stable two-pad grasp
at runtime.


### v35 EE-Delta Pilot

v34 proved that a discrete gripper trigger and contact-close reflex still give
0% strict grasp: contact is too brief and poorly aligned. v35 keeps the VLA
observation contract clean, but changes the action representation to
`[tcp_dx, tcp_dy, tcp_dz, gripper]` and executes TCP deltas through IK at rollout.
Run the full pipeline with:

```bash
bash scripts/run_v35_ee_delta.sh
```

This creates symlinked converted datasets under `data_ee/`, builds storage-aware
sampled caches, trains `pick_v35_ee_delta`, and runs both random and fixed-set
evaluation.

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

The current stable approach is not v28-only training. v28 fit its dataset but
regressed rollout reaching, so v29 mixes v26, v27, and v28 while initializing
from the strong v26 base. v30 focuses the cache on descent/contact/close timing.
v31 adds 10D contact-aware proprio and contact-filtered DAgger: roll out the
current policy, keep episodes where the policy actually contacts the cube, and
train on IK expert corrections from those policy-visited states. v33 escalates
that idea by switching to scripted expert intervention after the policy reaches
near/contact states, then training from those successful recovery trajectories.

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


## DAgger Collection

DAgger records the policy's own visited states and stores IK expert correction
actions for those observations. The expert can use simulator state while
labeling; the learned policy still receives only images, robot proprio,
target-relative proprio, and contact bits.

The first unfiltered DAgger attempt was mostly approach frames and almost no
contact. The useful dataset is the contact-filtered one:

```bash
cd /home/aditya/bude_vla
mkdir -p logs

MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/collect_dagger_pick.py \
  --ckpt checkpoints/pick_v30_contact_timing/pick_v30_contact_timing_final.pt \
  --out data/pick_v31_dagger_contact \
  --num-episodes 250 \
  --max-attempts 2500 \
  --max-steps 900 \
  --state-dim 10 \
  --min-contact-frames 2 \
  --exec-first-only \
  --seed 131 2>&1 | tee logs/pick_v31_dagger_contact.log
```

This produced 250 written episodes, 224348 frames, 4472 any-contact frames, and
480 strict-grasp frames. That is the first DAgger set worth training on.

Train v31 from v30 after building caches for old and DAgger data. The old 9D
datasets are padded to the 10D layout during training; the v30 checkpoint
proprio layer is adapted from 9D to 10D at initialization.

```bash
PHASE_RANGES="0.04:0.20:0.35,0.20:0.50:0.45,0.50:1.00:0.20"

for root in pick_v26_unified pick_v27_precision pick_v28_depth_nudge_recovery; do
  MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/build_frame_cache.py \
    --data-root data/$root \
    --out-dir data/$root/cache_224_h4_v31_balanced10k \
    --max-frames 10000 \
    --n-history-frames 4 \
    --phase-ranges "$PHASE_RANGES"
done

MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/build_frame_cache.py \
  --data-root data/pick_v31_dagger_contact \
  --out-dir data/pick_v31_dagger_contact/cache_224_h4_contact24k \
  --max-frames 24000 \
  --n-history-frames 4 \
  --contact-prob 0.75 \
  --contact-jitter 8

C1=data/pick_v26_unified/cache_224_h4_v31_balanced10k
C2=data/pick_v27_precision/cache_224_h4_v31_balanced10k
C3=data/pick_v28_depth_nudge_recovery/cache_224_h4_v31_balanced10k
C4=data/pick_v31_dagger_contact/cache_224_h4_contact24k
FRAME_CACHE="$C1:$C2:$C3:$C4"

MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/train.py \
  --data-root data/pick_v26_unified \
  --data-root data/pick_v27_precision \
  --data-root data/pick_v28_depth_nudge_recovery \
  --data-root data/pick_v31_dagger_contact \
  --frame-cache "$FRAME_CACHE" \
  --task pick_v31_dagger_balanced \
  --init-from checkpoints/pick_v30_contact_timing/pick_v30_contact_timing_final.pt \
  --use-dinov2 \
  --img-size 224 \
  --chunk-size 16 \
  --n-history-frames 4 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --num-workers 2 \
  --n-steps 60000 \
  --save-every 10000 \
  --eval-every 0 \
  --lr 2e-5 \
  --backbone-lr 1e-6 \
  --bc-loss-weight 6.0 \
  --flow-loss-weight 0.10 \
  --gripper-loss-weight 12.0 \
  --early-bc-weight 7.0 \
  --early-bc-frac 0.24 \
  --late-bc-weight 10.0 \
  --late-bc-frac 0.35 \
  --ema-decay 0.999 2>&1 | tee logs/pick_v31_dagger_balanced.log
```

## v33 Intervention DAgger

The v33 run uses `scripts/collect_dagger_pick.py --intervention-mode`. The
policy drives first; once it reaches near/contact with the cube, the proven
scripted pick expert takes over and demonstrates recovery, grasp, lift, and
place from that policy-visited state. This keeps cube position out of the
learned policy input while using simulator state only for expert labeling.

Actual v33 collection/training results from local logs:

```text
data/pick_v33_intervention_dagger
wrote: 1480/1500 episodes from 2500 attempts
successful expert-intervention episodes: 1480
frames: 1108605
state_dim: 10
any_contact frames: 602236 (0.5432)
strict_grasp frames: 570369 (0.5145)
training: 80000 steps, final_loss=0.013787
checkpoint: checkpoints/pick_v33_intervention_dagger/pick_v33_intervention_dagger_final.pt
```

Autonomous eval after training:

```text
random benchmark: 0/150 success
any_contact episodes: 45/150 (0.300)
strict_grasp episodes: 0/150
fixed 8-position eval: 0/8 success
```

Interpretation: v33 proves the dataset can contain stable grasp/place
corrections, but the autonomous policy still fails to close/alignment precisely
enough to produce strict two-pad contact. The next diagnostic is the opt-in
robot-side `--contact-close-reflex` in eval/benchmark. It uses only finger-pad
contact, not cube coordinates. If that creates strict grasps, close timing is
the blocker; if not, final millimeter alignment is still the blocker.

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
- Prefer visual/contact recovery data, DAgger, and loss weighting over privileged cube-position inputs.

## Inspiration

Built with reference to small VLA and robot imitation-learning ideas from X-VLA, SmolVLA, Octo, LeRobot-style datasets, flow matching, and ACT-style temporal ensembling.

## Citation

Built by Aditya, 2026.
