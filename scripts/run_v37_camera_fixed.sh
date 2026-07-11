#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/aditya/bude_vla
PYTHON=/home/aditya/venv-bude/bin/python
DATA_ROOT="$ROOT/data/pick_v37_camera_fixed"
CACHE_DIR="$DATA_ROOT/cache_224_h2_v37_6k"
TASK=pick_v37_camera_fixed
CKPT_DIR="$ROOT/checkpoints/$TASK"
LOG_DIR="$ROOT/logs"
VIDEO_DIR="$ROOT/demos/videos"

export MUJOCO_GL=egl
export PYTHONPATH="$ROOT/src"
export TMPDIR="$ROOT/.tmp"
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$LOG_DIR" "$VIDEO_DIR" "$TMPDIR" "$ROOT/data" "$ROOT/checkpoints"
cd "$ROOT"

stage() {
  printf '\n=== [%s] %s ===\n' "$1" "$2"
}

require_free_gb() {
  local required_gb=$1
  local available_kb
  local available_gb
  available_kb=$(df -Pk "$ROOT" | awk 'NR == 2 {print $4}')
  available_gb=$((available_kb / 1024 / 1024))
  echo "disk available: ${available_gb} GiB (required: ${required_gb} GiB)"
  if (( available_gb < required_gb )); then
    echo "FATAL: insufficient free disk; refusing to risk another full-disk desktop failure"
    exit 1
  fi
}

require_available_ram_gb() {
  local required_gb=$1
  local available_kb
  local available_gb
  available_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  available_gb=$((available_kb / 1024 / 1024))
  echo "RAM available: ${available_gb} GiB (required: ${required_gb} GiB)"
  if (( available_gb < required_gb )); then
    echo "FATAL: too little available RAM; close memory-heavy programs and rerun"
    exit 1
  fi
}

if [[ ! -x "$PYTHON" ]]; then
  echo "FATAL: Python environment not found at $PYTHON"
  exit 1
fi

if pgrep -af 'scripts/(train|record_pick_episodes|build_frame_cache)\.py' >/dev/null; then
  echo "FATAL: another BUD-E data/training process is already running"
  pgrep -af 'scripts/(train|record_pick_episodes|build_frame_cache)\.py'
  exit 1
fi

require_free_gb 45
require_available_ram_gb 5

stage 1/7 "Camera-only expert acceptance test"
"$PYTHON" scripts/benchmark_visual_servo_pick.py \
  --num-episodes 100 \
  --max-steps 2200 \
  --max-grasp-retries 2 \
  --min-success-rate 0.95 \
  --seed 777 2>&1 | tee "$LOG_DIR/pick_v37_visual_expert_bench.log"

stage 2/7 "Fresh camera-correct demonstrations"
if [[ -e "$DATA_ROOT" && ! -f "$DATA_ROOT/.record_complete" ]]; then
  echo "FATAL: partial dataset exists at $DATA_ROOT"
  echo "Inspect it, then remove that directory before restarting this fresh run."
  exit 1
fi

if [[ ! -f "$DATA_ROOT/.record_complete" ]]; then
  "$PYTHON" scripts/record_pick_episodes.py \
    --out "$DATA_ROOT" \
    --max-eps 4000 \
    --max-steps 2200 \
    --img-size 224 \
    --record-stride 4 \
    --state-dim 6 \
    --max-grasp-retries 1 \
    --recovery-jitter-xy 0.003 \
    --recovery-jitter-z 0.002 \
    --recovery-jitter-prob 0.20 \
    --nudge-recovery-prob 0.05 \
    --nudge-recovery-xy 0.003 \
    --nudge-recovery-z 0.002 \
    --retry-miss-xy 0.004 \
    --retry-miss-prob 0.08 \
    --seed 3701 2>&1 | tee "$LOG_DIR/pick_v37_record.log"

  N_EP=$(find "$DATA_ROOT/meta/episodes_index" -name 'episode_*.json' -type f | wc -l)
  echo "episodes written: $N_EP"
  if (( N_EP < 3200 )); then
    echo "FATAL: fewer than 3200 successful demonstrations; refusing to train"
    exit 1
  fi
  touch "$DATA_ROOT/.record_complete"
else
  echo "recording stage already complete; reusing $DATA_ROOT"
fi

require_free_gb 20

stage 3/7 "Persisted-action replay gate"
"$PYTHON" scripts/validate_dataset_replay.py \
  --data-root "$DATA_ROOT" \
  --num-episodes 200 \
  --seed 3702 \
  --min-success-rate 0.95 2>&1 | tee "$LOG_DIR/pick_v37_replay.log"

stage 4/7 "Memory-safe 224px frame cache"
if [[ -e "$CACHE_DIR" && ! -f "$CACHE_DIR/.cache_complete" ]]; then
  echo "FATAL: partial cache exists at $CACHE_DIR"
  echo "Remove only that cache directory before restarting."
  exit 1
fi

if [[ ! -f "$CACHE_DIR/.cache_complete" ]]; then
  "$PYTHON" scripts/build_frame_cache.py \
    --data-root "$DATA_ROOT" \
    --out-dir "$CACHE_DIR" \
    --max-frames 6000 \
    --n-history-frames 2 \
    --phase-ranges '0.00:0.25:3,0.25:0.60:3,0.60:1.00:2' \
    --seed 3703 2>&1 | tee "$LOG_DIR/pick_v37_cache.log"
  test -s "$CACHE_DIR/images.uint8.npy"
  test -s "$CACHE_DIR/global_indices.npy"
  touch "$CACHE_DIR/.cache_complete"
else
  echo "cache stage already complete; reusing $CACHE_DIR"
fi

require_free_gb 15
require_available_ram_gb 5

stage 5/7 "Fresh v37 training with closed-loop checkpoint selection"
if [[ ! -f "$CKPT_DIR/${TASK}_final.pt" ]]; then
  "$PYTHON" scripts/train.py \
    --data-root "$DATA_ROOT" \
    --frame-cache "$CACHE_DIR" \
    --task "$TASK" \
    --use-dinov2 \
    --img-size 224 \
    --chunk-size 16 \
    --n-history-frames 2 \
    --batch-size 4 \
    --grad-accum-steps 8 \
    --num-workers 1 \
    --n-steps 25000 \
    --save-every 5000 \
    --eval-every 5000 \
    --eval-episodes 12 \
    --eval-max-steps 400 \
    --eval-max-tries 1 \
    --eval-seed 3704 \
    --lr 1e-4 \
    --backbone-lr 1e-6 \
    --bc-loss-weight 8.0 \
    --flow-loss-weight 0.10 \
    --gripper-loss-weight 5.0 \
    --early-bc-weight 4.0 \
    --early-bc-frac 0.25 \
    --late-bc-weight 4.0 \
    --late-bc-frac 0.35 \
    --ema-decay 0.999 2>&1 | tee "$LOG_DIR/pick_v37_train.log"
else
  echo "training stage already complete; reusing $CKPT_DIR/${TASK}_final.pt"
fi

SELECTED_CKPT="$CKPT_DIR/${TASK}_best.pt"
if [[ ! -f "$SELECTED_CKPT" ]]; then
  SELECTED_CKPT="$CKPT_DIR/${TASK}_final.pt"
fi
if [[ ! -f "$SELECTED_CKPT" ]]; then
  echo "FATAL: no trained checkpoint found"
  exit 1
fi
printf '%s\n' "$SELECTED_CKPT" | tee "$LOG_DIR/pick_v37_selected_checkpoint.txt"

stage 6/7 "150-position learned-policy benchmark"
"$PYTHON" scripts/benchmark_random_pick.py \
  --ckpt "$SELECTED_CKPT" \
  --num-episodes 150 \
  --max-steps 450 \
  --seed 3705 \
  --ensembling \
  --ensembling-k 0.25 \
  --replan-every 1 2>&1 | tee "$LOG_DIR/pick_v37_random_bench.log"

stage 7/7 "Fixed-set diagnostic video"
"$PYTHON" scripts/eval_pick_ball.py \
  --ckpt "$SELECTED_CKPT" \
  --num-episodes 8 \
  --max-steps 450 \
  --ensembling \
  --ensembling-k 0.25 \
  --replan-every 1 \
  --cube-positions '0.23,-0.02;0.25,0.00;0.27,0.02;0.29,0.04;0.31,-0.01;0.33,0.05;0.22,0.06;0.34,0.03' \
  --out "$VIDEO_DIR/eval_pick_v37_camera_fixed.mp4" 2>&1 | tee "$LOG_DIR/pick_v37_video.log"

echo
echo "=== V37 PIPELINE COMPLETE ==="
echo "selected checkpoint: $SELECTED_CKPT"
echo "benchmark log: $LOG_DIR/pick_v37_random_bench.log"
echo "video: $VIDEO_DIR/eval_pick_v37_camera_fixed.mp4"
df -h "$ROOT"
