#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/aditya/bude_vla
PYTHON=/home/aditya/venv-bude/bin/python
SOURCE_ROOT="$ROOT/data/pick_v37_camera_fixed"
DATA_ROOT="$ROOT/data/pick_v41_ee_abs"
CACHE_DIR="$SOURCE_ROOT/cache_224_h2_v38_64k"
TASK=pick_v41_ee_abs
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

mkdir -p "$LOG_DIR" "$VIDEO_DIR" "$TMPDIR" "$ROOT/data" "$ROOT/checkpoints" "$CKPT_DIR"
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

if pgrep -af 'scripts/(train|record_pick_episodes|build_frame_cache|convert_dataset_to_ee_delta)\.py' >/dev/null; then
  echo "FATAL: another BUD-E data/training process is already running"
  pgrep -af 'scripts/(train|record_pick_episodes|build_frame_cache|convert_dataset_to_ee_delta)\.py'
  exit 1
fi

require_free_gb 20
require_available_ram_gb 5

stage 1/8 "Camera-only expert acceptance test"
"$PYTHON" scripts/benchmark_visual_servo_pick.py \
  --num-episodes 100 \
  --max-steps 2200 \
  --max-grasp-retries 2 \
  --min-success-rate 0.95 \
  --seed 777 2>&1 | tee "$LOG_DIR/pick_v41_visual_expert_bench.log"

stage 2/8 "Source demonstration and cache integrity"
test -s "$SOURCE_ROOT/meta/info.json"
test -s "$CACHE_DIR/images.uint8.npy"
test -s "$CACHE_DIR/global_indices.npy"
SOURCE_EPISODES=$(find "$SOURCE_ROOT/meta/episodes_index" -name "episode_*.json" -type f | wc -l)
echo "source episodes: $SOURCE_EPISODES"
if (( SOURCE_EPISODES < 3200 )); then
  echo "FATAL: source dataset has fewer than 3200 verified demonstrations"
  exit 1
fi

stage 3/8 "Relabel joint controls as absolute camera-conditioned TCP targets"
if [[ -e "$DATA_ROOT" && ! -f "$DATA_ROOT/meta/info.json" ]]; then
  echo "FATAL: partial task-space dataset exists at $DATA_ROOT"
  exit 1
fi
if [[ ! -f "$DATA_ROOT/meta/info.json" ]]; then
  "$PYTHON" scripts/convert_dataset_to_ee_delta.py \
    --src "$SOURCE_ROOT" \
    --out "$DATA_ROOT" \
    --action-space ee_abs \
    --lookahead-steps 0 2>&1 | tee "$LOG_DIR/pick_v41_convert.log"
else
  echo "task-space relabeling already complete; reusing $DATA_ROOT"
fi

ACTION_SPACE=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1])).get(\"action_space\"))" "$DATA_ROOT/meta/info.json")
TARGET_EPISODES=$(find "$DATA_ROOT/meta/episodes_index" -name "episode_*.json" -type f | wc -l)
echo "target episodes: $TARGET_EPISODES action_space=$ACTION_SPACE"
if [[ "$ACTION_SPACE" != "ee_abs" || "$TARGET_EPISODES" -ne "$SOURCE_EPISODES" ]]; then
  echo "FATAL: task-space relabeling is incomplete or has the wrong action space"
  exit 1
fi

stage 4/8 "Task-space persisted-action replay gate"
"$PYTHON" scripts/validate_dataset_replay.py \
  --data-root "$DATA_ROOT" \
  --num-episodes 200 \
  --seed 4102 \
  --min-success-rate 0.95 2>&1 | tee "$LOG_DIR/pick_v41_replay.log"

require_free_gb 15
require_available_ram_gb 5

stage 5/8 "Task-space VLA training with deterministic workspace selection"
if [[ ! -f "$CKPT_DIR/${TASK}_final.pt" ]]; then
  TRAIN_START_ARGS=(
    --init-from "$ROOT/checkpoints/pick_v40_radial_precision/pick_v40_radial_precision_best.pt"
    --init-from-raw
  )
  LATEST_STEP_CKPT=$(find "$CKPT_DIR" -maxdepth 1 -type f -name "${TASK}_step_*.pt" -print 2>/dev/null | sort -V | tail -n 1)
  if [[ -n "$LATEST_STEP_CKPT" ]]; then
    echo "resuming interrupted v41 run from $LATEST_STEP_CKPT"
    TRAIN_START_ARGS=(--resume "$LATEST_STEP_CKPT")
  fi

  "$PYTHON" scripts/train.py \
    --data-root "$DATA_ROOT" \
    --frame-cache "$CACHE_DIR" \
    --task "$TASK" \
    "${TRAIN_START_ARGS[@]}" \
    --allow-action-head-mismatch \
    --action-space ee_abs \
    --use-dinov2 \
    --img-size 224 \
    --chunk-size 16 \
    --n-history-frames 2 \
    --batch-size 4 \
    --grad-accum-steps 8 \
    --num-workers 0 \
    --n-steps 120000 \
    --save-every 5000 \
    --eval-every 5000 \
    --eval-episodes 36 \
    --eval-max-steps 450 \
    --eval-max-tries 1 \
    --seed 4103 \
    --eval-seed 4104 \
    --eval-grid-size 6 \
    --lr 3e-5 \
    --backbone-lr 1e-7 \
    --bc-loss-weight 8.0 \
    --flow-loss-weight 0.0 \
    --gripper-loss-weight 5.0 \
    --early-bc-weight 6.0 \
    --early-bc-frac 0.25 \
    --late-bc-weight 4.0 \
    --late-bc-frac 0.35 \
    --ema-decay 0 2>&1 | tee "$LOG_DIR/pick_v41_train.log"
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
printf "%s\n" "$SELECTED_CKPT" | tee "$LOG_DIR/pick_v41_selected_checkpoint.txt"

stage 6/8 "Task-space visual sensitivity diagnostic"
"$PYTHON" scripts/diag_action_sensitivity.py \
  --ckpt "$SELECTED_CKPT" \
  --cube 0.22,-0.03 \
  --cube 0.26,-0.03 \
  --cube 0.30,-0.03 \
  --cube 0.34,-0.03 \
  --cube 0.22,0.00 \
  --cube 0.26,0.00 \
  --cube 0.30,0.00 \
  --cube 0.34,0.00 \
  --cube 0.22,0.03 \
  --cube 0.26,0.03 \
  --cube 0.30,0.03 \
  --cube 0.34,0.03 \
  --cube 0.22,0.06 \
  --cube 0.26,0.06 \
  --cube 0.30,0.06 \
  --cube 0.34,0.06 2>&1 | tee "$LOG_DIR/pick_v41_task_space_sensitivity.log"

echo "comparing native chunks against first-action closed-loop replanning"
"$PYTHON" scripts/benchmark_random_pick.py \
  --ckpt "$SELECTED_CKPT" \
  --num-episodes 36 \
  --max-steps 450 \
  --seed 4110 \
  --raw-weights 2>&1 | tee "$LOG_DIR/pick_v41_mode_native.log"
"$PYTHON" scripts/benchmark_random_pick.py \
  --ckpt "$SELECTED_CKPT" \
  --num-episodes 36 \
  --max-steps 450 \
  --seed 4110 \
  --raw-weights \
  --exec-first-only 2>&1 | tee "$LOG_DIR/pick_v41_mode_first.log"

NATIVE_SUCCESS=$(awk "/^success episodes:/ {split(\$3, a, \"/\"); print a[1]}" "$LOG_DIR/pick_v41_mode_native.log")
FIRST_SUCCESS=$(awk "/^success episodes:/ {split(\$3, a, \"/\"); print a[1]}" "$LOG_DIR/pick_v41_mode_first.log")
EXEC_ARGS=()
EXEC_MODE=native_chunk
if (( FIRST_SUCCESS > NATIVE_SUCCESS )); then
  EXEC_ARGS=(--exec-first-only)
  EXEC_MODE=first_action
fi
echo "execution mode: $EXEC_MODE (native=$NATIVE_SUCCESS/36 first=$FIRST_SUCCESS/36)" | tee "$LOG_DIR/pick_v41_execution_mode.txt"

stage 7/8 "200-position learned-policy acceptance benchmark"
set +e
"$PYTHON" scripts/benchmark_random_pick.py \
  --ckpt "$SELECTED_CKPT" \
  --num-episodes 200 \
  --max-steps 450 \
  --seed 4105 \
  --raw-weights \
  "${EXEC_ARGS[@]}" \
  --min-success-rate 0.80 2>&1 | tee "$LOG_DIR/pick_v41_random_bench.log"
BENCH_STATUS=${PIPESTATUS[0]}
set -e

stage 8/8 "Fixed-set diagnostic video"
"$PYTHON" scripts/eval_pick_ball.py \
  --ckpt "$SELECTED_CKPT" \
  --num-episodes 8 \
  --max-steps 450 \
  --raw-weights \
  "${EXEC_ARGS[@]}" \
  --cube-positions "0.23,-0.02;0.25,0.00;0.27,0.02;0.29,0.04;0.31,-0.01;0.33,0.05;0.22,0.06;0.34,0.03" \
  --out "$VIDEO_DIR/eval_pick_v41_ee_abs.mp4" 2>&1 | tee "$LOG_DIR/pick_v41_video.log"

if (( BENCH_STATUS != 0 )); then
  echo "V41 did not meet the 80% acceptance gate; preserving step checkpoints for diagnosis."
  exit "$BENCH_STATUS"
fi

rm -f "$CKPT_DIR"/"${TASK}"_step_*.pt

echo
echo "=== V41 EE-ABS PIPELINE PASSED >=80% ==="
echo "selected checkpoint: $SELECTED_CKPT"
echo "benchmark log: $LOG_DIR/pick_v41_random_bench.log"
echo "video: $VIDEO_DIR/eval_pick_v41_ee_abs.mp4"
df -h "$ROOT"
