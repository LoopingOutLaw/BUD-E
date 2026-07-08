# Pick VLA Training Notes

This note records the current working direction for the SO-101 pick-and-place VLA experiments. It is intentionally practical: what failed, what changed, what is working now, and which commands reproduce the current training path.

## Core Constraint

The policy must not receive cube position as an oracle input. The scripted demonstrator may use simulator state to produce demonstrations, but the learned policy observes only:

- dual camera images,
- robot proprioception,
- language instruction,
- contact/proprio signals available to the robot-side policy interface.

This keeps the setup aligned with eventual real-time robot deployment, where the arm will not be handed privileged cube coordinates.

## Failure History

Early runs trained for many steps while producing 0 percent closed-loop success. The recurring rollout symptoms were:

- moving to a fixed pose regardless of cube position,
- touching or hovering near the cube without grasping,
- closing too early and pushing the cube away,
- freezing after a near miss.

Root causes found during debugging:

- success-only demonstrations did not teach recovery from small final-centimeter errors,
- gripper supervision was too weak relative to arm-joint dimensions,
- progress/time proprio acted as a shortcut and leaked demo timing into inference,
- v28-style recovery data by itself could fit supervised loss while regressing live reaching,
- final contact/close timing needed more focused samples than generic phase-balanced caches provided.

## Current Code Path

Important changes now in the repo:

- DINOv2 vision at 224 px with dual-camera history stacking.
- Visual/perception-conditioned action decoding.
- No progress/time shortcut in policy input.
- Recovery-jitter demonstrations for XY and Z descent errors.
- Retry demonstrations for missed grasps.
- Nudge/backoff demonstrations for light cube contact followed by recovery.
- Mixed cached training support for multiple data roots.
- Gripper-weighted BC and flow losses.
- `scripts/build_frame_cache.py --phase-ranges` for contact-focused cache sampling.
- 10D contact-aware proprio: base6 + target_rel2 + any_pad_contact + strict is_grasping.
- DAgger collection in `scripts/collect_dagger_pick.py`: policy rollouts labeled by an IK correction expert.
- Eval debug mode printing raw arm targets, clipping state, and gripper command.

## Experiment Milestones

### v26 unified

`pick_v26_unified` was the first useful base. It removed fixed-pose collapse and made the arm visually reach toward the cube almost every rollout.

Useful checkpoint:

```text
checkpoints/pick_v26_unified/pick_v26_unified_final.pt
checkpoints/pick_v26_unified/pick_v26_unified_step_060000.pt
```

### v27 precision

`pick_v27_precision` added cleaner precision and depth-recovery data. It improved local fitting but still froze around contact in some videos.

### v28 depth/nudge recovery

`pick_v28_depth_nudge_recovery` fit its own dataset, but v28-only rollout regressed: it could move less reliably toward the cube. The lesson was that recovery-only fine-tuning can overpower the base reaching skill.

### v29 mixed reach/precision/recovery

`pick_v29_mixed_reach_precision_recovery` mixed v26, v27, and v28 data while initializing from v26. This restored visual reaching and added retry-like behavior. Observed behavior:

- reliably moves toward the cube,
- descends near the cube,
- sometimes re-attempts after a failed grasp or pushed cube,
- still fails at terminal close/depth timing.

The debug eval showed late close timing in one representative case:

```text
step 0050 grip=+1.058  open above/near cube
step 0100 grip=+0.872  still open
step 0150 grip=+0.064  barely closing
step 0200 grip=-0.306  finally closing after the cube was disturbed
```

### v30 contact timing

`pick_v30_contact_timing` is the current focused fine-tune. It starts from v29 and trains on compact contact-focused caches. The goal is not generic extra training; it is to emphasize descend/contact/close/recovery frames.

### v31 DAgger round 1

`pick_v31_dagger_round1` is the next planned run. It differs from v28-v30 because the failure states are not hand-guessed. The v30 policy is rolled out in simulation, the current observation is recorded, and an IK expert labels the corrective action from that actual policy-visited state. This directly targets the terminal off-distribution states where the policy reaches, touches, pushes, or hovers but does not grasp.

## v29 Mixed Training

Use this when rebuilding the current base from v26/v27/v28:

```bash
cd /home/aditya/bude_vla
mkdir -p logs

FRAME_CACHE="data/pick_v26_unified/cache_224_h4_phase12k:data/pick_v27_precision/cache_224_h4_phase12k:data/pick_v28_depth_nudge_recovery/cache_224_h4_phase12k"

MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/train.py \
  --data-root data/pick_v26_unified \
  --data-root data/pick_v27_precision \
  --data-root data/pick_v28_depth_nudge_recovery \
  --frame-cache "$FRAME_CACHE" \
  --task pick_v29_mixed_reach_precision_recovery \
  --init-from checkpoints/pick_v26_unified/pick_v26_unified_final.pt \
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
  --lr 5e-5 \
  --backbone-lr 2e-6 \
  --bc-loss-weight 5.0 \
  --flow-loss-weight 0.15 \
  --gripper-loss-weight 8.0 \
  --early-bc-weight 8.0 \
  --early-bc-frac 0.25 \
  --late-bc-weight 12.0 \
  --late-bc-frac 0.42 \
  --ema-decay 0.999
```

## v30 Contact-Focused Fine-Tune

Build smaller caches biased toward the contact/close region:

```bash
cd /home/aditya/bude_vla
PHASE_RANGES="0.06:0.20:0.40,0.20:0.42:0.45,0.42:0.70:0.12,0.70:1.00:0.03"

MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/build_frame_cache.py \
  --data-root data/pick_v26_unified \
  --out-dir data/pick_v26_unified/cache_224_h4_contact8k \
  --max-frames 8000 \
  --n-history-frames 4 \
  --phase-ranges "$PHASE_RANGES"

MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/build_frame_cache.py \
  --data-root data/pick_v27_precision \
  --out-dir data/pick_v27_precision/cache_224_h4_contact8k \
  --max-frames 8000 \
  --n-history-frames 4 \
  --phase-ranges "$PHASE_RANGES"

MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/build_frame_cache.py \
  --data-root data/pick_v28_depth_nudge_recovery \
  --out-dir data/pick_v28_depth_nudge_recovery/cache_224_h4_contact8k \
  --max-frames 8000 \
  --n-history-frames 4 \
  --phase-ranges "$PHASE_RANGES"
```

Train v30 from v29:

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


## v31 DAgger Round 1

Collect policy-visited states from v30 and label them with the IK expert:

```bash
cd /home/aditya/bude_vla
mkdir -p logs

MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/collect_dagger_pick.py \
  --ckpt checkpoints/pick_v30_contact_timing/pick_v30_contact_timing_final.pt \
  --out data/pick_v31_dagger_round1 \
  --num-episodes 500 \
  --max-steps 700 \
  --state-dim 10 \
  --exec-first-only \
  --seed 131 2>&1 | tee logs/pick_v31_dagger_collect.log
```

Build compact caches. Old 9D roots are retained for reaching stability and padded to 10D by the training dataset reader. The DAgger root is native 10D.

```bash
PHASE_RANGES="0.04:0.20:0.35,0.20:0.50:0.45,0.50:1.00:0.20"

for root in pick_v26_unified pick_v27_precision pick_v28_depth_nudge_recovery; do
  MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/build_frame_cache.py \
    --data-root data/$root \
    --out-dir data/$root/cache_224_h4_v31_6k \
    --max-frames 6000 \
    --n-history-frames 4 \
    --phase-ranges "$PHASE_RANGES"
done

MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/build_frame_cache.py \
  --data-root data/pick_v31_dagger_round1 \
  --out-dir data/pick_v31_dagger_round1/cache_224_h4_v31_24k \
  --max-frames 24000 \
  --n-history-frames 4 \
  --phase-bins 8
```

Train the 10D v31 model from the 9D v30 checkpoint. `train.py` adapts the proprio input layer by splitting the legacy strict-grasp column across the new any-contact and strict-grasp columns.

```bash
FRAME_CACHE="data/pick_v26_unified/cache_224_h4_v31_6k:data/pick_v27_precision/cache_224_h4_v31_6k:data/pick_v28_depth_nudge_recovery/cache_224_h4_v31_6k:data/pick_v31_dagger_round1/cache_224_h4_v31_24k"

MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/train.py \
  --data-root data/pick_v26_unified \
  --data-root data/pick_v27_precision \
  --data-root data/pick_v28_depth_nudge_recovery \
  --data-root data/pick_v31_dagger_round1 \
  --frame-cache "$FRAME_CACHE" \
  --task pick_v31_dagger_round1 \
  --init-from checkpoints/pick_v30_contact_timing/pick_v30_contact_timing_final.pt \
  --use-dinov2 \
  --img-size 224 \
  --chunk-size 16 \
  --n-history-frames 4 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --num-workers 2 \
  --n-steps 50000 \
  --save-every 10000 \
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
  --ema-decay 0.999 2>&1 | tee logs/pick_v31_dagger_round1.log
```

## Evaluation

Use fixed cube positions for comparable videos:

```bash
MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/eval_pick_ball.py \
  --ckpt checkpoints/pick_v30_contact_timing/pick_v30_contact_timing_final.pt \
  --num-episodes 8 \
  --max-steps 1800 \
  --exec-first-only \
  --cube-positions '0.25,0.00;0.30,-0.04;0.30,0.06;0.22,0.05;0.28,-0.08;0.34,0.03;0.18,-0.04;0.31,0.08' \
  --out demos/videos/eval_pick_v30_final_firstonly.mp4
```

Smoother eval:

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

One-episode debug eval for close timing:

```bash
MUJOCO_GL=egl PYTHONPATH=src /home/aditya/venv-bude/bin/python scripts/eval_pick_ball.py \
  --ckpt checkpoints/pick_v30_contact_timing/pick_v30_contact_timing_final.pt \
  --num-episodes 1 \
  --max-steps 1800 \
  --exec-first-only \
  --debug-actions \
  --cube-positions '0.30,0.06' \
  --out demos/videos/eval_pick_v30_debug_one.mp4 2>&1 | tee logs/eval_pick_v30_debug_one.log
```

Interpretation:

- `grip` stays positive/open at contact: close timing is late.
- `grip` goes negative/closed but cube slips: depth/contact alignment is wrong.
- `clip=True`: the policy is asking for unreachable joint targets. Recent debug runs showed `clip=False`, so the known issue is learned terminal behavior, not joint clipping.

## Operational Notes

- Keep `--eval-every 0` during training on the laptop; integrated eval can spike RAM.
- Use cached frames for speed; use smaller contact caches when targeting terminal grasp behavior.
- Keep raw datasets and final checkpoints before deleting caches. Caches are reproducible and can be rebuilt.
- If a command is accidentally started twice, stop one run and verify which output directory is complete before continuing.
- If a model reaches the cube but fails to grasp, do not add privileged cube position. Diagnose gripper timing, depth alignment, contact recovery, and DAgger correction data instead.
