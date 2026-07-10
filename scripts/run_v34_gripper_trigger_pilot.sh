#!/bin/bash
set -e
set -o pipefail

cd /home/aditya/bude_vla
source /home/aditya/venv-bude/bin/activate 2>/dev/null || true
export MUJOCO_GL=egl
export PYTHONPATH=src
mkdir -p logs

echo "=== [0/5] Clean stale v34 pilot outputs ==="
rm -rf \
  data/pick_v34_touch_close_pilot \
  checkpoints/pick_v34_gripper_trigger_pilot

echo "=== [1/5] Collect v34 touch-close DAgger pilot ==="
python scripts/collect_dagger_pick.py \
  --ckpt checkpoints/pick_v33_intervention_dagger/pick_v33_intervention_dagger_final.pt \
  --out data/pick_v34_touch_close_pilot \
  --num-episodes 150 \
  --max-attempts 700 \
  --max-steps 1200 \
  --state-dim 10 \
  --intervention-mode \
  --intervention-trigger touch \
  --intervention-steps 280 \
  --max-interventions 1 \
  --min-contact-frames 10 \
  --min-grasp-frames 20 \
  --exec-first-only \
  --seed 434 2>&1 | tee logs/pick_v34_touch_close_collect.log

echo "=== [2/5] QC v34 pilot data ==="
python - <<'PY'
import glob
import numpy as np
import pyarrow.parquet as pq
files = glob.glob("data/pick_v34_touch_close_pilot/data/chunk-*/episode_*.parquet")
states, actions = [], []
for f in files:
    tab = pq.read_table(f, columns=["observation.state", "action"])
    states.extend(tab.column("observation.state").to_pylist())
    actions.extend(tab.column("action").to_pylist())
states = np.asarray(states, dtype=np.float32)
actions = np.asarray(actions, dtype=np.float32)
print("episodes:", len(files))
print("frames:", len(states))
print("state_dim:", states.shape[1] if len(states) else None)
print("any_contact frames:", int((states[:, 8] > 0.5).sum()) if len(states) else 0)
print("strict_grasp frames:", int((states[:, 9] > 0.5).sum()) if len(states) else 0)
print("close_action frames:", int((actions[:, -1] <= 0.0).sum()) if len(actions) else 0)
print("close_action frac:", float((actions[:, -1] <= 0.0).mean()) if len(actions) else 0.0)
PY

echo "=== [3/5] Build pilot caches ==="
PHASE_RANGES="0.04:0.20:0.40,0.20:0.50:0.40,0.50:1.00:0.20"
for root in pick_v26_unified pick_v27_precision pick_v28_depth_nudge_recovery; do
  python scripts/build_frame_cache.py \
    --data-root data/$root \
    --out-dir data/$root/cache_224_h4_v34_base4k \
    --max-frames 4000 \
    --n-history-frames 4 \
    --phase-ranges "$PHASE_RANGES"
done

python scripts/build_frame_cache.py \
  --data-root data/pick_v31_dagger_contact \
  --out-dir data/pick_v31_dagger_contact/cache_224_h4_v34_contact8k \
  --max-frames 8000 \
  --n-history-frames 4 \
  --contact-prob 0.75 \
  --contact-jitter 8

python scripts/build_frame_cache.py \
  --data-root data/pick_v33_intervention_dagger \
  --out-dir data/pick_v33_intervention_dagger/cache_224_h4_v34_intervention12k \
  --max-frames 12000 \
  --n-history-frames 4 \
  --contact-prob 0.80 \
  --contact-jitter 10

python scripts/build_frame_cache.py \
  --data-root data/pick_v34_touch_close_pilot \
  --out-dir data/pick_v34_touch_close_pilot/cache_224_h4_touch_close16k \
  --max-frames 16000 \
  --n-history-frames 4 \
  --contact-prob 0.90 \
  --contact-jitter 6

echo "=== [4/5] Train v34 pilot with discrete gripper trigger head ==="
C1=data/pick_v26_unified/cache_224_h4_v34_base4k
C2=data/pick_v27_precision/cache_224_h4_v34_base4k
C3=data/pick_v28_depth_nudge_recovery/cache_224_h4_v34_base4k
C4=data/pick_v31_dagger_contact/cache_224_h4_v34_contact8k
C5=data/pick_v33_intervention_dagger/cache_224_h4_v34_intervention12k
C6=data/pick_v34_touch_close_pilot/cache_224_h4_touch_close16k
FRAME_CACHE="$C1:$C2:$C3:$C4:$C5:$C6"

python scripts/train.py \
  --data-root data/pick_v26_unified \
  --data-root data/pick_v27_precision \
  --data-root data/pick_v28_depth_nudge_recovery \
  --data-root data/pick_v31_dagger_contact \
  --data-root data/pick_v33_intervention_dagger \
  --data-root data/pick_v34_touch_close_pilot \
  --frame-cache "$FRAME_CACHE" \
  --task pick_v34_gripper_trigger_pilot \
  --init-from checkpoints/pick_v33_intervention_dagger/pick_v33_intervention_dagger_final.pt \
  --use-dinov2 \
  --img-size 224 \
  --chunk-size 16 \
  --n-history-frames 4 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --num-workers 2 \
  --n-steps 20000 \
  --save-every 5000 \
  --eval-every 0 \
  --lr 1e-5 \
  --backbone-lr 5e-7 \
  --bc-loss-weight 8.0 \
  --flow-loss-weight 0.05 \
  --gripper-loss-weight 18.0 \
  --use-gripper-trigger-head \
  --gripper-trigger-loss-weight 4.0 \
  --gripper-trigger-label-threshold 0.0 \
  --gripper-trigger-threshold 0.45 \
  --gripper-trigger-close-value -1.0 \
  --early-bc-weight 4.0 \
  --early-bc-frac 0.20 \
  --late-bc-weight 18.0 \
  --late-bc-frac 0.30 \
  --ema-decay 0.999 2>&1 | tee logs/pick_v34_gripper_trigger_pilot_train.log

echo "=== [5/5] Pilot random benchmark ==="
python scripts/benchmark_random_pick.py \
  --ckpt checkpoints/pick_v34_gripper_trigger_pilot/pick_v34_gripper_trigger_pilot_final.pt \
  --num-episodes 60 \
  --max-steps 1200 \
  --exec-first-only \
  --seed 734 2>&1 | tee logs/pick_v34_gripper_trigger_pilot_bench60.log

echo "=== V34 PILOT DONE ==="
