#!/usr/bin/env bash
set -euo pipefail
cd /home/aditya/bude_vla
source /home/aditya/venv-bude/bin/activate 2>/dev/null || true
export MUJOCO_GL=egl
export PYTHONPATH=src
mkdir -p logs data_ee

echo "=== [0/5] Disk before v35 ==="
df -h /home/aditya/bude_vla

echo "=== [1/5] Convert joint-action datasets to EE-delta actions ==="
python scripts/convert_dataset_to_ee_delta.py --src data/pick_v26_unified --out data_ee/pick_v26_unified_ee --overwrite
python scripts/convert_dataset_to_ee_delta.py --src data/pick_v27_precision --out data_ee/pick_v27_precision_ee --overwrite
python scripts/convert_dataset_to_ee_delta.py --src data/pick_v28_depth_nudge_recovery --out data_ee/pick_v28_depth_nudge_recovery_ee --overwrite
python scripts/convert_dataset_to_ee_delta.py --src data/pick_v34_touch_close_pilot --out data_ee/pick_v34_touch_close_pilot_ee --overwrite

echo "=== [2/5] Build modest frame caches that fit current storage ==="
rm -rf \
  data_ee/pick_v26_unified_ee/cache_224_h4_v35_3k \
  data_ee/pick_v27_precision_ee/cache_224_h4_v35_3k \
  data_ee/pick_v28_depth_nudge_recovery_ee/cache_224_h4_v35_4k \
  data_ee/pick_v34_touch_close_pilot_ee/cache_224_h4_v35_8k
python scripts/build_frame_cache.py --data-root data_ee/pick_v26_unified_ee --out-dir data_ee/pick_v26_unified_ee/cache_224_h4_v35_3k --max-frames 3000 --n-history-frames 4 --phase-ranges '0.00:0.28:4,0.28:0.55:2,0.55:1.00:1' --seed 351
python scripts/build_frame_cache.py --data-root data_ee/pick_v27_precision_ee --out-dir data_ee/pick_v27_precision_ee/cache_224_h4_v35_3k --max-frames 3000 --n-history-frames 4 --phase-ranges '0.00:0.28:4,0.28:0.55:2,0.55:1.00:1' --seed 352
python scripts/build_frame_cache.py --data-root data_ee/pick_v28_depth_nudge_recovery_ee --out-dir data_ee/pick_v28_depth_nudge_recovery_ee/cache_224_h4_v35_4k --max-frames 4000 --n-history-frames 4 --phase-ranges '0.00:0.30:3,0.30:0.62:3,0.62:1.00:2' --seed 353
python scripts/build_frame_cache.py --data-root data_ee/pick_v34_touch_close_pilot_ee --out-dir data_ee/pick_v34_touch_close_pilot_ee/cache_224_h4_v35_8k --max-frames 8000 --n-history-frames 4 --contact-prob 0.70 --contact-jitter 10 --phase-ranges '0.00:0.35:1,0.35:1.00:5' --seed 354

echo "=== [3/5] Train v35 EE-delta policy ==="
FRAME_CACHE="data_ee/pick_v26_unified_ee/cache_224_h4_v35_3k:data_ee/pick_v27_precision_ee/cache_224_h4_v35_3k:data_ee/pick_v28_depth_nudge_recovery_ee/cache_224_h4_v35_4k:data_ee/pick_v34_touch_close_pilot_ee/cache_224_h4_v35_8k"
python scripts/train.py \
  --data-root data_ee/pick_v26_unified_ee \
  --data-root data_ee/pick_v27_precision_ee \
  --data-root data_ee/pick_v28_depth_nudge_recovery_ee \
  --data-root data_ee/pick_v34_touch_close_pilot_ee \
  --frame-cache "$FRAME_CACHE" \
  --task pick_v35_ee_delta \
  --init-from checkpoints/pick_v34_gripper_trigger_pilot/pick_v34_gripper_trigger_pilot_final.pt \
  --allow-action-head-mismatch \
  --action-space ee_delta \
  --ee-delta-scale 0.05 \
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
  --lr 4e-5 \
  --backbone-lr 1e-6 \
  --bc-loss-weight 8.0 \
  --flow-loss-weight 0.05 \
  --gripper-loss-weight 10.0 \
  --early-bc-weight 8.0 \
  --early-bc-frac 0.28 \
  --late-bc-weight 10.0 \
  --late-bc-frac 0.38 \
  --use-gripper-trigger-head \
  --gripper-trigger-loss-weight 2.0 \
  --gripper-trigger-label-threshold 0.0 \
  --gripper-trigger-threshold 0.45 \
  --gripper-trigger-close-value -1.0 \
  --ema-decay 0.999 2>&1 | tee logs/pick_v35_ee_delta_train.log

echo "=== [4/5] Random benchmark ==="
python scripts/benchmark_random_pick.py \
  --ckpt checkpoints/pick_v35_ee_delta/pick_v35_ee_delta_final.pt \
  --num-episodes 150 \
  --max-steps 1000 \
  --exec-first-only \
  --seed 535 2>&1 | tee logs/pick_v35_ee_delta_bench.log

echo "=== [5/5] Fixed-set video ==="
python scripts/eval_pick_ball.py \
  --ckpt checkpoints/pick_v35_ee_delta/pick_v35_ee_delta_final.pt \
  --num-episodes 8 \
  --max-steps 1800 \
  --exec-first-only \
  --cube-positions '0.25,0.00;0.30,-0.04;0.30,0.06;0.22,0.05;0.28,-0.08;0.34,0.03;0.18,-0.04;0.31,0.08' \
  --out demos/videos/eval_pick_v35_ee_delta_final.mp4 2>&1 | tee logs/pick_v35_ee_delta_video.log

echo "=== V35 EE-DELTA DONE ==="
df -h /home/aditya/bude_vla
