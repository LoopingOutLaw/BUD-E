#!/usr/bin/env bash
# Eval watch - polls for new ckpts, runs multi_seed_eval as soon as one appears
# Usage: bash scripts/run_eval_watch.sh [--inits 5] [--rollouts 10] [--quit-after 1]
set -eu
cd "$(dirname "$0")/.."

unset PYTHONPATH
export PYTHONPATH=src

INITS=5
ROLLOUTS=10
QUIT_AFTER=1   # stop after N evaluations
SEEN_FINAL_NPZ=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --inits) INITS=$2; shift 2;;
    --rollouts) ROLLOUTS=$2; shift 2;;
    --quit-after) QUIT_AFTER=$2; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

echo "[eval-watch] inits=$INITS  rollouts=$ROLLOUTS  quit_after=$QUIT_AFTER"

DONE=0
while [[ $DONE -lt $QUIT_AFTER ]]; do
  # wait for any ckpt newer than the seed file
  LATEST=$(ls -t checkpoints/pick_v4_25k/pick_v4_25k_step_*.pt 2>/dev/null | head -1 || true)
  if [[ -z "$LATEST" ]]; then
    echo "[eval-watch] no checkpoints yet, sleeping 60s"
    sleep 60
    continue
  fi
  STEP=$(basename "$LATEST" | sed 's/.*step_0*\([0-9]*\)\.pt/\1/')
  MARKER="results/.eval_done_step_${STEP}"
  if [[ -f "$MARKER" ]]; then
    echo "[eval-watch] step $STEP already evaluated, sleeping 60s"
    sleep 60
    continue
  fi
  # wait for training to finish writing (size stable)
  SIZE1=$(stat -c%s "$LATEST")
  sleep 4
  SIZE2=$(stat -c%s "$LATEST")
  if [[ $SIZE1 -ne $SIZE2 ]]; then
    echo "[eval-watch] $LATEST still growing ($SIZE1 -> $SIZE2), waiting"
    sleep 10
    continue
  fi
  echo "[eval-watch] evaluating $LATEST at $(date +%T)"
  mkdir -p results logs
  /home/aditya/.bude-venv/bin/python scripts/multi_seed_eval.py \
    --ckpt "$LATEST" --inits $INITS --rollouts $ROLLOUTS --ckpt-suffix "step_${STEP}" \
    2>&1 | tee "logs/eval_step_${STEP}.log" | tail -25
  RESULT=$?
  echo "$RESULT" > "$MARKER"
  echo "[eval-watch] step $STEP eval result code=$RESULT"
  DONE=$((DONE+1))
done
echo "[eval-watch] quota of $QUIT_AFTER eval(s) reached, exiting"
