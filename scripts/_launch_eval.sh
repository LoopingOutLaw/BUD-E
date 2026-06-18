#!/usr/bin/env bash
# Robust detached eval launcher.
# Usage: bash scripts/_launch_eval.sh "cmd and args" /path/to/log
set -eu
CMD="$1"
LOG="$2"
cd /home/aditya/bude_vla
unset PYTHONPATH
export MUJOCO_GL=egl
export PYTHONPATH=src
mkdir -p "$(dirname "$LOG")"
setsid bash -c "$CMD" >>"$LOG" 2>&1 < /dev/null &
echo $! > /tmp/eval_pid.txt
disown
echo "spawned pid=$(cat /tmp/eval_pid.txt), log=$LOG"
