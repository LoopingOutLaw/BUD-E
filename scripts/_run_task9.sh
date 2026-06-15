#!/bin/bash
cd /home/aditya/bude_vla
unset PYTHONPATH
export MUJOCO_GL=egl
export PYTHONPATH=src
exec /home/aditya/.bude-venv/bin/python scripts/rollout_policy.py \
  --ckpt /home/aditya/bude_vla/checkpoints/pick_224/pick_224_final.pt \
  --out /home/aditya/bude_vla/demos/videos/pick_vla_rollout.mp4 \
  --num-rollouts 5 --img-size 224 --max-tries 3 --max-steps-per-try 350 --seed 42
