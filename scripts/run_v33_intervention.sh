#!/bin/bash
set -e
set -o pipefail

cd /home/aditya/bude_vla
source /home/aditya/venv-bude/bin/activate 2>/dev/null || true
export MUJOCO_GL=egl
export PYTHONPATH=src

echo "=== [1/6] Collect v33 scripted-intervention DAgger data ==="
python scripts/collect_dagger_pick.py \
  --ckpt checkpoints/pick_v31_dagger_balanced/pick_v31_dagger_balanced_final.pt \
  --out data/pick_v33_intervention_dagger \
  --num-episodes 1500 \
  --max-attempts 2500 \
  --max-steps 1800 \
  --state-dim 10 \
  --intervention-mode \
  --intervention-steps 1600 \
  --near-trigger-dist 0.075 \
  --min-contact-frames 10 \
  --min-grasp-frames 20 \
  --require-success \
  --exec-first-only \
  --seed 333 2>&1 | tee logs/pick_v33_intervention_collect.log

echo "=== [2/6] QC collected dataset ==="
python - <<'PY'
import glob
import numpy as np
import pyarrow.parquet as pq

files = glob.glob("data/pick_v33_intervention_dagger/data/chunk-*/episode_*.parquet")
states = []
for f in files:
    states.extend(pq.read_table(f, columns=["observation.state"]).column("observation.state").to_pylist())
states = np.asarray(states, dtype=np.float32)
print("episodes:", len(files))
print("frames:", len(states))
print("state_dim:", states.shape[1])
print("any_contact frames:", int((states[:, 8] > 0.5).sum()))
print("any_contact frac:", float((states[:, 8] > 0.5).mean()))
print("strict_grasp frames:", int((states[:, 9] > 0.5).sum()))
print("strict_grasp frac:", float((states[:, 9] > 0.5).mean()))
PY

echo "=== [3/6] Build caches ==="
PHASE_RANGES="0.04:0.20:0.35,0.20:0.50:0.45,0.50:1.00:0.20"

for root in pick_v26_unified pick_v27_precision pick_v28_depth_nudge_recovery; do
  python scripts/build_frame_cache.py \
    --data-root data/$root \
    --out-dir data/$root/cache_224_h4_v33_base8k \
    --max-frames 8000 \
    --n-history-frames 4 \
    --phase-ranges "$PHASE_RANGES"
done

python scripts/build_frame_cache.py \
  --data-root data/pick_v31_dagger_contact \
  --out-dir data/pick_v31_dagger_contact/cache_224_h4_v33_contact12k \
  --max-frames 12000 \
  --n-history-frames 4 \
  --contact-prob 0.75 \
  --contact-jitter 8

python scripts/build_frame_cache.py \
  --data-root data/pick_v33_intervention_dagger \
  --out-dir data/pick_v33_intervention_dagger/cache_224_h4_intervention40k \
  --max-frames 40000 \
  --n-history-frames 4 \
  --contact-prob 0.85 \
  --contact-jitter 10

echo "=== [4/6] Train v33 from v31, not v32 ==="
C1=data/pick_v26_unified/cache_224_h4_v33_base8k
C2=data/pick_v27_precision/cache_224_h4_v33_base8k
C3=data/pick_v28_depth_nudge_recovery/cache_224_h4_v33_base8k
C4=data/pick_v31_dagger_contact/cache_224_h4_v33_contact12k
C5=data/pick_v33_intervention_dagger/cache_224_h4_intervention40k
FRAME_CACHE="$C1:$C2:$C3:$C4:$C5"

python scripts/train.py \
  --data-root data/pick_v26_unified \
  --data-root data/pick_v27_precision \
  --data-root data/pick_v28_depth_nudge_recovery \
  --data-root data/pick_v31_dagger_contact \
  --data-root data/pick_v33_intervention_dagger \
  --frame-cache "$FRAME_CACHE" \
  --task pick_v33_intervention_dagger \
  --init-from checkpoints/pick_v31_dagger_balanced/pick_v31_dagger_balanced_final.pt \
  --use-dinov2 \
  --img-size 224 \
  --chunk-size 16 \
  --n-history-frames 4 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --num-workers 2 \
  --n-steps 80000 \
  --save-every 10000 \
  --eval-every 0 \
  --lr 1e-5 \
  --backbone-lr 5e-7 \
  --bc-loss-weight 8.0 \
  --flow-loss-weight 0.05 \
  --gripper-loss-weight 18.0 \
  --early-bc-weight 4.0 \
  --early-bc-frac 0.20 \
  --late-bc-weight 18.0 \
  --late-bc-frac 0.30 \
  --ema-decay 0.999 2>&1 | tee logs/pick_v33_intervention_train.log

echo "=== [5/6] Random benchmark ==="
python scripts/benchmark_random_pick.py \
  --ckpt checkpoints/pick_v33_intervention_dagger/pick_v33_intervention_dagger_final.pt \
  --num-episodes 150 \
  --max-steps 1200 \
  --exec-first-only \
  --seed 733 2>&1 | tee logs/pick_v33_random_bench.log

echo "=== [6/6] Fixed video eval ==="
python scripts/eval_pick_ball.py \
  --ckpt checkpoints/pick_v33_intervention_dagger/pick_v33_intervention_dagger_final.pt \
  --num-episodes 8 \
  --max-steps 1800 \
  --exec-first-only \
  --cube-positions '0.25,0.00;0.30,-0.04;0.30,0.06;0.22,0.05;0.28,-0.08;0.34,0.03;0.18,-0.04;0.31,0.08' \
  --out demos/videos/eval_pick_v33_intervention_firstonly.mp4 2>&1 | tee logs/pick_v33_video.log

echo "=== V33 PIPELINE DONE ==="
