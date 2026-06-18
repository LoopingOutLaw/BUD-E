#!/bin/bash
# Wait for first checkpoint, then run quick eval and append result to log
while true; do
    CKPT=$(ls -1 /home/aditya/bude_vla/checkpoints/pick_v4_25k/*.pt 2>/dev/null | tail -1)
    if [ -n "$CKPT" ]; then
        echo "[$(date)] Found checkpoint: $CKPT" >> /home/aditya/bude_vla/logs/auto_eval_result.log
        cd /home/aditya/bude_vla && unset PYTHONPATH; MUJOCO_GL=egl PYTHONPATH=src \
            /home/aditya/.bude-venv/bin/python scripts/quick_eval.py \
            >> /home/aditya/bude_vla/logs/auto_eval_result.log 2>&1
        echo "---" >> /home/aditya/bude_vla/logs/auto_eval_result.log
        break
    fi
    sleep 30
done
