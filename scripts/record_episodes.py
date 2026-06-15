"""Script to record 500 reach + 500 push episodes into LeRobot v3 format.

Uses CPU-only MuJoCo recorder (no JAX GPU allocation) to avoid OOM on 8GB GPUs.
"""
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

from bude_vla.data.cpu_recorder import record_dataset_cpu

t0 = time.time()

print("=== Recording 500 reach episodes ===")
record_dataset_cpu(task="reach", n_episodes=500, n_steps=30,
                   root="/home/aditya/bude_vla/data/reach_v3")
print(f"  reach done in {time.time()-t0:.0f}s")

print("=== Recording 500 push episodes ===")
record_dataset_cpu(task="push", n_episodes=500, n_steps=40,
                   root="/home/aditya/bude_vla/data/push_v3")
print(f"  push done in {time.time()-t0:.0f}s")

print("=== DONE ===")
