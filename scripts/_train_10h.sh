#!/bin/bash
set -euo pipefail

# 10-hour overnight BUD-E VLA pick training (11-dim proprio + augment).
# Dataset: 1,000 episodes (49,959 frames) at 224x224, proprio = arm(8) + cube_xyz(3).
# Steps:   170,000 (~10h at 4.8 sps)
# Saves:   every 10k steps + final.

cd /home/aditya/bude_vla
unset PYTHONPATH
export MUJOCO_GL=egl
export PYTHONPATH=src
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p /home/aditya/bude_vla/logs
LOG=/home/aditya/bude_vla/logs/train_10h.log

echo "=== 10h training start at $(date) ===" >> "$LOG"

exec /home/aditya/.bude-venv/bin/python -u scripts/train.py \
  --n-steps 170000 \
  --batch-size 32 \
  --chunk-size 4 \
  --img-size 224 \
  --lr 3e-4 \
  --save-every 10000 \
  --augment \
  --data-root /home/aditya/bude_vla/data/pick_v3_224_prop11 \
  --task pick_224_10h
