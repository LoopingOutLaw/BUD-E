#!/usr/bin/env bash
set -euo pipefail
cd /home/aditya/bude_vla
source /home/aditya/venv-bude/bin/activate 2>/dev/null || true
export MUJOCO_GL=egl
export PYTHONPATH=src
mkdir -p logs data_ee

echo "=== [0/5] Disk before v36 ==="
df -h /home/aditya/bude_vla

echo "=== [1/5] Convert to LOOKAHEAD EE-delta labels ==="
python scripts/convert_dataset_to_ee_delta.py --src data/pick_v26_unified --out data_ee/pick_v26_unified_ee_l12 --lookahead-steps 12 --max-delta 0.10 --overwrite
python scripts/convert_dataset_to_ee_delta.py --src data/pick_v27_precision --out data_ee/pick_v27_precision_ee_l12 --lookahead-steps 12 --max-delta 0.10 --overwrite
python scripts/convert_dataset_to_ee_delta.py --src data/pick_v28_depth_nudge_recovery --out data_ee/pick_v28_depth_nudge_recovery_ee_l12 --lookahead-steps 12 --max-delta 0.10 --overwrite
python scripts/convert_dataset_to_ee_delta.py --src data/pick_v34_touch_close_pilot --out data_ee/pick_v34_touch_close_pilot_ee_l12 --lookahead-steps 12 --max-delta 0.10 --overwrite

echo "=== [2/5] Build storage-aware frame caches ==="
rm -rf \
  data_ee/pick_v26_unified_ee_l12/cache_224_h4_v36_3k \
  data_ee/pick_v27_precision_ee_l12/cache_224_h4_v36_3k \
  data_ee/pick_v28_depth_nudge_recovery_ee_l12/cache_224_h4_v36_4k \
  data_ee/pick_v34_touch_close_pilot_ee_l12/cache_224_h4_v36_8k
python scripts/build_frame_cache.py --data-root data_ee/pick_v26_unified_ee_l12 --out-dir data_ee/pick_v26_unified_ee_l12/cache_224_h4_v36_3k --max-frames 3000 --n-history-frames 4 --phase-ranges '0.00:0.30:5,0.30:0.58:2,0.58:1.00:1' --seed 361
python scripts/build_frame_cache.py --data-root data_ee/pick_v27_precision_ee_l12 --out-dir data_ee/pick_v27_precision_ee_l12/cache_224_h4_v36_3k --max-frames 3000 --n-history-frames 4 --phase-ranges '0.00:0.30:5,0.30:0.58:2,0.58:1.00:1' --seed 362
python scripts/build_frame_cache.py --data-root data_ee/pick_v28_depth_nudge_recovery_ee_l12 --out-dir data_ee/pick_v28_depth_nudge_recovery_ee_l12/cache_224_h4_v36_4k --max-frames 4000 --n-history-frames 4 --phase-ranges '0.00:0.32:4,0.32:0.65:3,0.65:1.00:2' --seed 363
python scripts/build_frame_cache.py --data-root data_ee/pick_v34_touch_close_pilot_ee_l12 --out-dir data_ee/pick_v34_touch_close_pilot_ee_l12/cache_224_h4_v36_8k --max-frames 8000 --n-history-frames 4 --contact-prob 0.70 --contact-jitter 10 --phase-ranges '0.00:0.35:1,0.35:1.00:5' --seed 364

echo "=== [3/5] Train v36 lookahead EE-delta policy ==="
FRAME_CACHE="data_ee/pick_v26_unified_ee_l12/cache_224_h4_v36_3k:data_ee/pick_v27_precision_ee_l12/cache_224_h4_v36_3k:data_ee/pick_v28_depth_nudge_recovery_ee_l12/cache_224_h4_v36_4k:data_ee/pick_v34_touch_close_pilot_ee_l12/cache_224_h4_v36_8k"
python scripts/train.py \
  --data-root data_ee/pick_v26_unified_ee_l12 \
  --data-root data_ee/pick_v27_precision_ee_l12 \
  --data-root data_ee/pick_v28_depth_nudge_recovery_ee_l12 \
  --data-root data_ee/pick_v34_touch_close_pilot_ee_l12 \
  --frame-cache "$FRAME_CACHE" \
  --task pick_v36_ee_delta_lookahead \
  --init-from checkpoints/pick_v34_gripper_trigger_pilot/pick_v34_gripper_trigger_pilot_final.pt \
  --allow-action-head-mismatch \
  --action-space ee_delta \
  --ee-delta-scale 0.08 \
  --use-dinov2 \
  --img-size 224 \
  --chunk-size 16 \
  --n-history-frames 4 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --num-workers 2 \
  --n-steps 70000 \
  --save-every 10000 \
  --eval-every 0 \
  --lr 5e-5 \
  --backbone-lr 1e-6 \
  --bc-loss-weight 10.0 \
  --flow-loss-weight 0.05 \
  --gripper-loss-weight 10.0 \
  --early-bc-weight 10.0 \
  --early-bc-frac 0.30 \
  --late-bc-weight 8.0 \
  --late-bc-frac 0.40 \
  --use-gripper-trigger-head \
  --gripper-trigger-loss-weight 2.0 \
  --gripper-trigger-label-threshold 0.0 \
  --gripper-trigger-threshold 0.45 \
  --gripper-trigger-close-value -1.0 \
  --ema-decay 0.999 2>&1 | tee logs/pick_v36_ee_delta_lookahead_train.log

echo "=== [4/5] Random benchmark ==="
python scripts/benchmark_random_pick.py \
  --ckpt checkpoints/pick_v36_ee_delta_lookahead/pick_v36_ee_delta_lookahead_final.pt \
  --num-episodes 150 \
  --max-steps 1000 \
  --exec-first-only \
  --seed 636 2>&1 | tee logs/pick_v36_ee_delta_lookahead_bench.log

echo "=== [5/5] Fixed-set video ==="
python scripts/eval_pick_ball.py \
  --ckpt checkpoints/pick_v36_ee_delta_lookahead/pick_v36_ee_delta_lookahead_final.pt \
  --num-episodes 8 \
  --max-steps 1800 \
  --exec-first-only \
  --cube-positions '0.25,0.00;0.30,-0.04;0.30,0.06;0.22,0.05;0.28,-0.08;0.34,0.03;0.18,-0.04;0.31,0.08' \
  --out demos/videos/eval_pick_v36_ee_delta_lookahead_final.mp4 2>&1 | tee logs/pick_v36_ee_delta_lookahead_video.log

echo "=== V36 EE-DELTA LOOKAHEAD DONE ==="
df -h /home/aditya/bude_vla
