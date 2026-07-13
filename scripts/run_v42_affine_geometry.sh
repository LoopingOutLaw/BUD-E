#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/aditya/bude_vla
PYTHON=/home/aditya/venv-bude/bin/python
DATA_ROOT="$ROOT/data/pick_v41_ee_abs"
CACHE_DIR="$ROOT/data/pick_v37_camera_fixed/cache_224_h2_v38_64k"
INIT_CKPT="$ROOT/checkpoints/pick_v41_ee_abs/pick_v41_ee_abs_best.pt"
TASK=pick_v42_affine_geometry
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

mkdir -p "$LOG_DIR" "$VIDEO_DIR" "$TMPDIR" "$CKPT_DIR"
cd "$ROOT"

stage() {
  printf '\n=== [%s] %s ===\n' "$1" "$2"
}

available_gb=$(df -Pk "$ROOT" | awk 'NR == 2 {print int($4 / 1024 / 1024)}')
available_ram_gb=$(awk '/MemAvailable:/ {print int($2 / 1024 / 1024)}' /proc/meminfo)
echo "disk available: ${available_gb} GiB; RAM available: ${available_ram_gb} GiB"
if (( available_gb < 18 )); then
  echo "FATAL: fewer than 18 GiB free; refusing to risk another full-disk failure"
  exit 1
fi
if (( available_ram_gb < 5 )); then
  echo "FATAL: fewer than 5 GiB RAM available; close memory-heavy applications"
  exit 1
fi
if pgrep -af 'scripts/(train|record_pick_episodes|build_frame_cache|convert_dataset_to_ee_delta)\.py' >/dev/null; then
  echo "FATAL: another BUD-E data/training process is active"
  pgrep -af 'scripts/(train|record_pick_episodes|build_frame_cache|convert_dataset_to_ee_delta)\.py'
  exit 1
fi

stage 1/7 "Contract and artifact checks"
test -s "$DATA_ROOT/meta/info.json"
test -s "$CACHE_DIR/images.uint8.npy"
test -s "$CACHE_DIR/global_indices.npy"
test -s "$INIT_CKPT"
"$PYTHON" - <<'PY'
import json
from pathlib import Path

info = json.loads(Path("data/pick_v41_ee_abs/meta/info.json").read_text())
assert info.get("action_space") == "ee_abs", info.get("action_space")
assert info["features"]["action"]["shape"] == [4]
print("dataset action contract: ee_abs [tcp_x, tcp_y, tcp_z, gripper]")
PY

stage 2/7 "Information-preserving task-space VLA training"
if [[ ! -f "$CKPT_DIR/${TASK}_final.pt" ]]; then
  TRAIN_START_ARGS=(
    --init-from "$INIT_CKPT"
    --init-from-raw
    --init-drop-prefix proprio.
    --init-drop-prefix perception_proj.
    --init-drop-prefix context_action_head.
  )
  latest_step=$(find "$CKPT_DIR" -maxdepth 1 -type f -name "${TASK}_step_*.pt" -print | sort -V | tail -n 1)
  if [[ -n "$latest_step" ]]; then
    echo "resuming interrupted v42 run from $latest_step"
    TRAIN_START_ARGS=(--resume "$latest_step")
  fi

  "$PYTHON" scripts/train.py \
    --data-root "$DATA_ROOT" \
    --frame-cache "$CACHE_DIR" \
    --task "$TASK" \
    "${TRAIN_START_ARGS[@]}" \
    --action-space ee_abs \
    --input-feature-norm affine \
    --direct-proprio-action-cond \
    --use-dinov2 \
    --img-size 224 \
    --chunk-size 16 \
    --n-history-frames 2 \
    --batch-size 4 \
    --grad-accum-steps 8 \
    --num-workers 0 \
    --n-steps 160000 \
    --save-every 5000 \
    --eval-every 5000 \
    --eval-episodes 36 \
    --eval-max-steps 450 \
    --eval-max-tries 1 \
    --seed 4203 \
    --eval-seed 4204 \
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
    --ema-decay 0.995 2>&1 | tee "$LOG_DIR/pick_v42_train.log"
else
  echo "training already complete; reusing $CKPT_DIR/${TASK}_final.pt"
fi

SELECTED_CKPT="$CKPT_DIR/${TASK}_best.pt"
if [[ ! -s "$SELECTED_CKPT" ]]; then
  SELECTED_CKPT="$CKPT_DIR/${TASK}_final.pt"
fi
test -s "$SELECTED_CKPT"
printf '%s\n' "$SELECTED_CKPT" | tee "$LOG_DIR/pick_v42_selected_checkpoint.txt"

stage 3/7 "Offline imitation and geometric sensitivity diagnostics"
"$PYTHON" scripts/diag_dataset_action_error.py \
  --ckpt "$SELECTED_CKPT" \
  --data-root "$DATA_ROOT" \
  --frame-cache "$CACHE_DIR" \
  --episodes 50 \
  --samples-per-episode 4 \
  --seed 4205 2>&1 | tee "$LOG_DIR/pick_v42_dataset_action_error.log"

"$PYTHON" scripts/diag_action_sensitivity.py \
  --ckpt "$SELECTED_CKPT" \
  --cube 0.22,-0.03 --cube 0.26,-0.03 --cube 0.30,-0.03 --cube 0.34,-0.03 \
  --cube 0.22,0.00  --cube 0.26,0.00  --cube 0.30,0.00  --cube 0.34,0.00 \
  --cube 0.22,0.03  --cube 0.26,0.03  --cube 0.30,0.03  --cube 0.34,0.03 \
  --cube 0.22,0.06  --cube 0.26,0.06  --cube 0.30,0.06  --cube 0.34,0.06 \
  2>&1 | tee "$LOG_DIR/pick_v42_task_space_sensitivity.log"

stage 4/7 "Paired deployment-mode selection"
BEST_SCORE=-1
BEST_MODE=unset
DEPLOY_ARGS=()

run_mode() {
  local name=$1
  shift
  local log="$LOG_DIR/pick_v42_mode_${name}.log"
  "$PYTHON" scripts/benchmark_random_pick.py \
    --ckpt "$SELECTED_CKPT" \
    --num-episodes 64 \
    --max-steps 450 \
    --seed 4210 \
    "$@" 2>&1 | tee "$log"
  local score
  score=$(awk '/^success episodes:/ {split($3, a, "/"); print a[1]}' "$log")
  if [[ -z "$score" ]]; then
    echo "FATAL: could not parse success count from $log"
    exit 1
  fi
  if (( score > BEST_SCORE )); then
    BEST_SCORE=$score
    BEST_MODE=$name
    DEPLOY_ARGS=("$@")
  fi
}

run_mode ema_h16
run_mode ema_h12 --execute-horizon 12
run_mode ema_h8 --execute-horizon 8
run_mode ema_ensemble4 --ensembling --ensembling-k 0.5 --replan-every 4
run_mode raw_h16 --raw-weights

{
  echo "mode=$BEST_MODE score=$BEST_SCORE/64"
  printf 'args='
  printf '%q ' "${DEPLOY_ARGS[@]}"
  printf '\n'
} | tee "$LOG_DIR/pick_v42_execution_mode.txt"

stage 5/7 "Two-hundred-position acceptance benchmark"
set +e
"$PYTHON" scripts/benchmark_random_pick.py \
  --ckpt "$SELECTED_CKPT" \
  --num-episodes 200 \
  --max-steps 450 \
  --seed 4206 \
  "${DEPLOY_ARGS[@]}" \
  --min-success-rate 0.80 2>&1 | tee "$LOG_DIR/pick_v42_random_bench.log"
BENCH_STATUS=${PIPESTATUS[0]}
set -e

stage 6/7 "Fixed-set diagnostic video"
"$PYTHON" scripts/eval_pick_ball.py \
  --ckpt "$SELECTED_CKPT" \
  --num-episodes 8 \
  --max-steps 450 \
  "${DEPLOY_ARGS[@]}" \
  --cube-positions "0.23,-0.02;0.25,0.00;0.27,0.02;0.29,0.04;0.31,-0.01;0.33,0.05;0.22,0.06;0.34,0.03" \
  --out "$VIDEO_DIR/eval_pick_v42_affine_geometry.mp4" \
  2>&1 | tee "$LOG_DIR/pick_v42_video.log"

stage 7/7 "Result"
if (( BENCH_STATUS != 0 )); then
  echo "V42 improved diagnostics are preserved, but the policy did not meet 80%."
  echo "Do not extend training blindly; use the printed workspace/stage breakdown."
  exit "$BENCH_STATUS"
fi

echo "V42 passed the 80% random-position acceptance gate."
echo "selected checkpoint: $SELECTED_CKPT"
echo "deployment mode: $BEST_MODE"
echo "benchmark: $LOG_DIR/pick_v42_random_bench.log"
echo "video: $VIDEO_DIR/eval_pick_v42_affine_geometry.mp4"
