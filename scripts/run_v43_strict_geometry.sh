#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
  elif [[ -x /home/aditya/venv-bude/bin/python ]]; then
    PYTHON=/home/aditya/venv-bude/bin/python
  else
    PYTHON=$(command -v python3 || true)
  fi
elif [[ "$PYTHON" != */* ]]; then
  PYTHON=$(command -v "$PYTHON" || true)
fi
JOINT_DATA_ROOT="$ROOT/data/pick_v43_strict_joint"
EE_DATA_ROOT="$ROOT/data/pick_v43_strict_ee_abs"
CACHE_DIR="$EE_DATA_ROOT/cache_224_h2_reset52k"
INIT_CKPT=${INIT_CKPT:-"$ROOT/checkpoints/pick_v42_affine_geometry/pick_v42_affine_geometry_best.pt"}
TASK=pick_v43_strict_geometry
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

mkdir -p "$ROOT/data" "$ROOT/checkpoints" "$LOG_DIR" "$VIDEO_DIR" "$TMPDIR"
cd "$ROOT"

stage() {
  printf '\n=== [%s] %s ===\n' "$1" "$2"
}

require_free_gb() {
  local required_gb=$1
  local available_gb
  available_gb=$(df -Pk "$ROOT" | awk 'NR == 2 {print int($4 / 1024 / 1024)}')
  echo "disk available: ${available_gb} GiB (required: ${required_gb} GiB)"
  if (( available_gb < required_gb )); then
    echo "FATAL: insufficient disk; refusing to risk another full-disk failure"
    exit 1
  fi
}

require_available_ram_gb() {
  local required_gb=$1
  local available_gb
  available_gb=$(awk '/MemAvailable:/ {print int($2 / 1024 / 1024)}' /proc/meminfo)
  echo "RAM available: ${available_gb} GiB (required: ${required_gb} GiB)"
  if (( available_gb < required_gb )); then
    echo "FATAL: close memory-heavy applications before running v43"
    exit 1
  fi
}

stage 0/10 "Preflight"
if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
  echo "FATAL: no Python interpreter found; activate a venv or set PYTHON"
  exit 1
fi
test -s "$INIT_CKPT"
require_free_gb 70
require_available_ram_gb 6
if pgrep -af 'scripts/(train|record_pick_episodes|build_frame_cache|convert_dataset_to_ee_delta)\.py' >/dev/null; then
  echo "FATAL: another BUD-E data or training process is active"
  pgrep -af 'scripts/(train|record_pick_episodes|build_frame_cache|convert_dataset_to_ee_delta)\.py'
  exit 1
fi

stage 1/10 "Strict expert gate with the exact recording recipe"
"$PYTHON" scripts/benchmark_scripted_pick.py \
  --num-episodes 100 \
  --max-steps 2200 \
  --max-grasp-retries 1 \
  --recovery-jitter-xy 0.003 \
  --recovery-jitter-z 0.002 \
  --recovery-jitter-prob 0.20 \
  --nudge-recovery-prob 0.05 \
  --nudge-recovery-xy 0.003 \
  --nudge-recovery-z 0.002 \
  --retry-miss-xy 0.004 \
  --retry-miss-prob 0.08 \
  --seed 4301 \
  --min-success-rate 0.98 2>&1 | tee "$LOG_DIR/pick_v43_expert_gate.log"

stage 2/10 "Fresh fixed-anchor demonstrations"
if [[ -e "$JOINT_DATA_ROOT" && ! -f "$JOINT_DATA_ROOT/.record_complete" ]]; then
  echo "FATAL: partial recording exists at $JOINT_DATA_ROOT"
  echo "Inspect it before removing that one directory and restarting."
  exit 1
fi
if [[ ! -f "$JOINT_DATA_ROOT/.record_complete" ]]; then
  "$PYTHON" scripts/record_pick_episodes.py \
    --out "$JOINT_DATA_ROOT" \
    --max-eps 5000 \
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
    --seed 4302 2>&1 | tee "$LOG_DIR/pick_v43_record.log"

  n_ep=$(find "$JOINT_DATA_ROOT/meta/episodes_index" -type f -name 'episode_*.json' | wc -l)
  echo "strict policy-rate demonstrations written: $n_ep"
  if (( n_ep < 4500 )); then
    echo "FATAL: fewer than 4500 replay-verified demonstrations"
    exit 1
  fi
  touch "$JOINT_DATA_ROOT/.record_complete"
else
  echo "recording already complete; reusing $JOINT_DATA_ROOT"
fi

stage 3/10 "Persisted joint-action replay gate"
"$PYTHON" scripts/validate_dataset_replay.py \
  --data-root "$JOINT_DATA_ROOT" \
  --num-episodes 250 \
  --seed 4303 \
  --min-success-rate 0.98 2>&1 | tee "$LOG_DIR/pick_v43_joint_replay.log"

stage 4/10 "Convert demonstrations to absolute task-space actions"
if [[ -e "$EE_DATA_ROOT" && ! -f "$EE_DATA_ROOT/.convert_complete" ]]; then
  echo "FATAL: partial conversion exists at $EE_DATA_ROOT"
  exit 1
fi
if [[ ! -f "$EE_DATA_ROOT/.convert_complete" ]]; then
  "$PYTHON" scripts/convert_dataset_to_ee_delta.py \
    --src "$JOINT_DATA_ROOT" \
    --out "$EE_DATA_ROOT" \
    --action-space ee_abs 2>&1 | tee "$LOG_DIR/pick_v43_convert.log"
  touch "$EE_DATA_ROOT/.convert_complete"
else
  echo "task-space conversion already complete; reusing $EE_DATA_ROOT"
fi

"$PYTHON" scripts/validate_dataset_replay.py \
  --data-root "$EE_DATA_ROOT" \
  --num-episodes 250 \
  --seed 4304 \
  --min-success-rate 0.95 2>&1 | tee "$LOG_DIR/pick_v43_ee_abs_replay.log"

stage 5/10 "Reset-anchored 52k dual-camera frame cache"
require_free_gb 45
if [[ -e "$CACHE_DIR" && ! -f "$CACHE_DIR/.cache_complete" ]]; then
  echo "FATAL: partial cache exists at $CACHE_DIR"
  exit 1
fi
if [[ ! -f "$CACHE_DIR/.cache_complete" ]]; then
  "$PYTHON" scripts/build_frame_cache.py \
    --data-root "$EE_DATA_ROOT" \
    --out-dir "$CACHE_DIR" \
    --max-frames 52000 \
    --n-history-frames 2 \
    --anchor-local-frames 0,1,2,4,8,16 \
    --min-frames-per-episode 4 \
    --phase-ranges '0.00:0.25:2,0.25:0.60:4,0.60:1.00:3' \
    --seed 4305 2>&1 | tee "$LOG_DIR/pick_v43_cache.log"
  test -s "$CACHE_DIR/images.uint8.npy"
  test -s "$CACHE_DIR/global_indices.npy"
  touch "$CACHE_DIR/.cache_complete"
else
  echo "cache already complete; reusing $CACHE_DIR"
fi

stage 6/10 "Fresh strict-geometry action decoder training"
require_free_gb 12
mkdir -p "$CKPT_DIR"
if [[ ! -f "$CKPT_DIR/${TASK}_final.pt" ]]; then
  train_start_args=(
    --init-from "$INIT_CKPT"
    --init-from-raw
    --init-drop-prefix context_action_head.
  )
  latest_step=$(find "$CKPT_DIR" -maxdepth 1 -type f -name "${TASK}_step_*.pt" -print | sort -V | tail -n 1)
  if [[ -n "$latest_step" ]]; then
    echo "resuming interrupted v43 run from $latest_step"
    train_start_args=(--resume "$latest_step")
  fi

  "$PYTHON" scripts/train.py \
    --data-root "$EE_DATA_ROOT" \
    --frame-cache "$CACHE_DIR" \
    --task "$TASK" \
    "${train_start_args[@]}" \
    --action-space ee_abs \
    --input-feature-norm affine \
    --direct-proprio-action-cond \
    --raw-geometry-action-cond \
    --use-dinov2 \
    --img-size 224 \
    --chunk-size 16 \
    --n-history-frames 2 \
    --batch-size 4 \
    --grad-accum-steps 8 \
    --num-workers 0 \
    --n-steps 220000 \
    --save-every 10000 \
    --keep-last-checkpoints 3 \
    --eval-every 10000 \
    --eval-episodes 64 \
    --eval-max-steps 450 \
    --eval-max-tries 1 \
    --eval-grid-size 8 \
    --seed 4306 \
    --eval-seed 4307 \
    --lr 2e-5 \
    --action-head-lr 1e-4 \
    --backbone-lr 1e-7 \
    --bc-loss-weight 8.0 \
    --bc-loss-type l1 \
    --chunk-end-bc-weight 4.0 \
    --flow-loss-weight 0.0 \
    --gripper-loss-weight 5.0 \
    --early-bc-weight 6.0 \
    --early-bc-frac 0.25 \
    --late-bc-weight 4.0 \
    --late-bc-frac 0.35 \
    --ema-decay 0 2>&1 | tee "$LOG_DIR/pick_v43_train.log"
else
  echo "training already complete; reusing $CKPT_DIR/${TASK}_final.pt"
fi

SELECTED_CKPT="$CKPT_DIR/${TASK}_best.pt"
if [[ ! -s "$SELECTED_CKPT" ]]; then
  SELECTED_CKPT="$CKPT_DIR/${TASK}_final.pt"
fi
test -s "$SELECTED_CKPT"
printf '%s\n' "$SELECTED_CKPT" | tee "$LOG_DIR/pick_v43_selected_checkpoint.txt"

stage 7/10 "Offline imitation and full-chunk geometry diagnostics"
"$PYTHON" scripts/diag_dataset_action_error.py \
  --ckpt "$SELECTED_CKPT" \
  --data-root "$EE_DATA_ROOT" \
  --frame-cache "$CACHE_DIR" \
  --episodes 50 \
  --samples-per-episode 4 \
  --seed 4308 \
  --raw-weights 2>&1 | tee "$LOG_DIR/pick_v43_dataset_action_error.log"

set +e
"$PYTHON" scripts/diag_action_sensitivity.py \
  --ckpt "$SELECTED_CKPT" \
  --raw-weights \
  --cube 0.22,-0.03 --cube 0.26,-0.03 --cube 0.30,-0.03 --cube 0.34,-0.03 \
  --cube 0.22,0.00  --cube 0.26,0.00  --cube 0.30,0.00  --cube 0.34,0.00 \
  --cube 0.22,0.03  --cube 0.26,0.03  --cube 0.30,0.03  --cube 0.34,0.03 \
  --cube 0.22,0.06  --cube 0.26,0.06  --cube 0.30,0.06  --cube 0.34,0.06 \
  --max-task-space-p95-mm 25 \
  2>&1 | tee "$LOG_DIR/pick_v43_chunk_geometry.log"
GEOMETRY_STATUS=${PIPESTATUS[0]}
set -e

stage 8/10 "Corrected deployment-mode selection"
BEST_SCORE=-1
BEST_MODE=unset
DEPLOY_ARGS=()

run_mode() {
  local name=$1
  shift
  local log="$LOG_DIR/pick_v43_mode_${name}.log"
  "$PYTHON" scripts/benchmark_random_pick.py \
    --ckpt "$SELECTED_CKPT" \
    --raw-weights \
    --num-episodes 64 \
    --max-steps 450 \
    --max-tries 1 \
    --seed 4310 \
    "$@" 2>&1 | tee "$log"
  local score
  score=$(awk '/^success episodes:/ {split($3, a, "/"); print a[1]}' "$log")
  if [[ -z "$score" ]]; then
    echo "FATAL: could not parse $log"
    exit 1
  fi
  if (( score > BEST_SCORE )); then
    BEST_SCORE=$score
    BEST_MODE=$name
    DEPLOY_ARGS=("$@")
  fi
}

run_mode full_chunk
run_mode horizon_8 --execute-horizon 8
run_mode horizon_4 --execute-horizon 4
run_mode first_only --exec-first-only
run_mode ensemble_4 --ensembling --ensembling-k 0.5 --replan-every 4

{
  echo "mode=$BEST_MODE score=$BEST_SCORE/64"
  printf 'args='
  printf '%q ' "${DEPLOY_ARGS[@]}"
  printf '\n'
} | tee "$LOG_DIR/pick_v43_execution_mode.txt"

stage 9/10 "Broad strict one-shot and local feedback-retry benchmarks"
"$PYTHON" scripts/benchmark_random_pick.py \
  --ckpt "$SELECTED_CKPT" \
  --raw-weights \
  --num-episodes 200 \
  --max-steps 450 \
  --max-tries 1 \
  --seed 4311 \
  "${DEPLOY_ARGS[@]}" 2>&1 | tee "$LOG_DIR/pick_v43_random_one_try.log"

set +e
"$PYTHON" scripts/benchmark_random_pick.py \
  --ckpt "$SELECTED_CKPT" \
  --raw-weights \
  --num-episodes 200 \
  --max-steps 650 \
  --max-tries 1 \
  --local-grasp-retry \
  --local-grasp-retries 2 \
  --seed 4311 \
  "${DEPLOY_ARGS[@]}" \
  --min-success-rate 0.80 2>&1 | tee "$LOG_DIR/pick_v43_local_retry_random200.log"
ACCEPT_STATUS=${PIPESTATUS[0]}
set -e

stage 10/10 "Strict-placement diagnostic video and result"
"$PYTHON" scripts/eval_pick_ball.py \
  --ckpt "$SELECTED_CKPT" \
  --raw-weights \
  --num-episodes 8 \
  --max-steps 650 \
  --max-tries 1 \
  --local-grasp-retry \
  --local-grasp-retries 2 \
  "${DEPLOY_ARGS[@]}" \
  --cube-positions '0.23,-0.02;0.25,0.00;0.27,0.02;0.29,0.04;0.31,-0.01;0.33,0.05;0.22,0.06;0.34,0.03' \
  --out "$VIDEO_DIR/eval_pick_v43_local_retry.mp4" \
  2>&1 | tee "$LOG_DIR/pick_v43_local_retry_video.log"

echo "selected checkpoint: $SELECTED_CKPT"
echo "deployment mode: $BEST_MODE"
echo "geometry gate status: $GEOMETRY_STATUS"
echo "one-try benchmark: $LOG_DIR/pick_v43_random_one_try.log"
echo "local-retry benchmark: $LOG_DIR/pick_v43_local_retry_random200.log"
echo "video: $VIDEO_DIR/eval_pick_v43_local_retry.mp4"
df -h "$ROOT"

if (( ACCEPT_STATUS != 0 )); then
  echo "V43 completed, but did not meet the strict 80% local-retry acceptance gate."
  exit "$ACCEPT_STATUS"
fi

echo "V43 passed the strict 80% random-position acceptance gate."
