"""Quick checkpoint inspector — no MuJoCo/model needed, just torch + the file.

Usage:
    python inspect_ckpt.py checkpoints/pick_v11_dinov2_chunk8/pick_v11_dinov2_chunk8_step_150000.pt
    python inspect_ckpt.py checkpoints/pick_v11_dinov2_chunk8/*.pt   # multiple ckpts, sorted by step
"""
import sys
import glob
import torch

paths = []
for arg in sys.argv[1:]:
    paths.extend(sorted(glob.glob(arg)))

if not paths:
    print("Usage: python inspect_ckpt.py <ckpt_path_or_glob> [...]")
    sys.exit(1)

rows = []
for p in paths:
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    step = ckpt.get("step", "?")
    cfg = ckpt.get("config", {})
    loss_hist = ckpt.get("loss_history", [])
    eval_hist = ckpt.get("eval_history", [])
    final_loss = loss_hist[-1][1] if loss_hist else float("nan")
    rows.append((step, p, cfg, final_loss, loss_hist, eval_hist))

rows.sort(key=lambda r: r[0] if isinstance(r[0], int) else -1)

for step, p, cfg, final_loss, loss_hist, eval_hist in rows:
    print(f"\n=== {p} ===")
    print(f"  step={step}  final_loss={final_loss:.6f}")
    print(f"  config: img_size={cfg.get('img_size')} use_dinov2={cfg.get('use_dinov2')} "
          f"chunk_size={cfg.get('chunk_size')} n_history_frames={cfg.get('n_history_frames')} "
          f"action_dim={cfg.get('action_dim')} state_dim={cfg.get('state_dim')} "
          f"dinov2_finetune_blocks={cfg.get('dinov2_finetune_blocks')} "
          f"bc={cfg.get('use_bc_head')} visual_cond={cfg.get('use_visual_action_cond')} "
          f"context={cfg.get('use_context_action_head')} perception={cfg.get('use_perception')} "
          f"perception_action_cond={cfg.get('use_perception_action_cond')}")
    if loss_hist:
        print(f"  loss: first={loss_hist[0]}  mid={loss_hist[len(loss_hist)//2]}  last={loss_hist[-1]}")
    else:
        print("  loss_history: EMPTY")
    if eval_hist:
        print(f"  eval_history ({len(eval_hist)} points): {eval_hist}")
    else:
        print("  eval_history: EMPTY  <-- you did not train this checkpoint with --eval-every")
