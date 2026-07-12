"""BUD-E training script.

Trains the BUDEPolicy on collected reach + push data with flow-matching loss.
Saves checkpoints and produces a single training-progress video.

Usage:
    PYTHONPATH=src python scripts/train.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import math
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler

from bude_vla.data.lerobot_v3 import BUDETrainingDataset
from bude_vla.models.policy import BUDEPolicy, BUDEConfig


def collate_fn(batch: list[dict]) -> dict:
    keys = batch[0].keys()
    out = {}
    for k in keys:
        v0 = batch[0][k]
        if isinstance(v0, str):
            # Leave language instruction as list[str] — never tensor-stack strings.
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def build_action_dim_weights(
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    gripper_loss_weight: float,
    shoulder_pan_loss_weight: float = 1.0,
    shoulder_lift_loss_weight: float = 1.0,
) -> torch.Tensor:
    dim_w = torch.ones(action_dim, device=device, dtype=dtype)
    if action_dim > 0:
        dim_w[0] = shoulder_pan_loss_weight
    if action_dim > 1:
        dim_w[1] = shoulder_lift_loss_weight
    if action_dim > 0:
        dim_w[-1] = gripper_loss_weight
    return dim_w


def build_bc_loss_weights(
    phase: torch.Tensor | None,
    mask: torch.Tensor,
    action_dim: int,
    early_bc_frac: float,
    early_bc_weight: float,
    late_bc_frac: float,
    late_bc_weight: float,
    gripper_loss_weight: float,
    shoulder_pan_loss_weight: float = 1.0,
    shoulder_lift_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sample/dimension weights and denominator for weighted BC loss."""
    if phase is None:
        sample_w = torch.ones((mask.shape[0], 1, 1), device=mask.device, dtype=mask.dtype)
    else:
        phase = phase.to(device=mask.device, dtype=mask.dtype)
        sample_w_1d = torch.ones_like(phase)
        if early_bc_weight != 1.0:
            sample_w_1d = torch.where(
                phase <= early_bc_frac,
                torch.full_like(sample_w_1d, early_bc_weight),
                sample_w_1d,
            )
        if late_bc_weight != 1.0:
            sample_w_1d = torch.where(
                phase >= late_bc_frac,
                torch.full_like(sample_w_1d, late_bc_weight),
                sample_w_1d,
            )
        sample_w = sample_w_1d.view(-1, 1, 1)

    dim_w = build_action_dim_weights(
        action_dim, mask.device, mask.dtype, gripper_loss_weight,
        shoulder_pan_loss_weight, shoulder_lift_loss_weight)
    dim_w_view = dim_w.view(1, 1, action_dim)
    denom = (mask.unsqueeze(-1) * sample_w * dim_w_view).sum().clamp_min(1.0)
    return sample_w, dim_w, denom


class PhaseBalancedBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Sample batches across episodes at similar phases.

    This is slower than single-episode batches, but it is better for visual
    grounding: each batch contains different cube positions at comparable task
    phases, so the model cannot minimize BC loss by predicting one average
    approach action. The lazy cache stays bounded by --lazy-cache-size.
    """

    def __init__(self, dataset: BUDETrainingDataset, batch_size: int,
                 seed: int = 0, early_prob: float = 0.5,
                 early_max_frac: float = 0.18):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.early_prob = float(early_prob)
        self.early_max_frac = float(early_max_frac)
        self.n_batches = max(1, len(dataset) // self.batch_size)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        ep_lengths = np.asarray([int(ep["length"]) for ep in self.dataset._episodes], dtype=np.int64)
        ep_weights = ep_lengths.astype(np.float64)
        ep_weights = ep_weights / ep_weights.sum()
        n_eps = len(ep_lengths)
        for _ in range(self.n_batches):
            if rng.random() < self.early_prob:
                phase = float(rng.uniform(0.0, self.early_max_frac))
            else:
                phase = float(rng.uniform(0.0, 1.0))
            replace = n_eps < self.batch_size
            ep_ids = rng.choice(n_eps, size=self.batch_size, replace=replace, p=ep_weights)
            batch = []
            for ep_i in ep_ids:
                length = int(ep_lengths[ep_i])
                # Small jitter avoids repeatedly sampling identical frame numbers.
                jitter = int(rng.integers(-3, 4))
                local = int(round(phase * max(0, length - 1))) + jitter
                local = min(max(local, 0), length - 1)
                batch.append(self.dataset._cum_frames[int(ep_i)] + local)
            yield batch


def _detect_dims(roots, key) -> list[int]:
    """Read dataset shapes for `key` from available info.json files."""
    dims: list[int] = []
    for root in roots:
        info_path = Path(root) / "meta" / "info.json"
        if info_path.exists():
            try:
                meta = json.loads(info_path.read_text())
                feat = meta.get("features", {}).get(key, {})
                shape = feat.get("shape", [])
                if isinstance(shape, list) and shape:
                    dims.append(int(shape[0]))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    return dims


def _detect_dim(roots, key):
    dims = _detect_dims(roots, key)
    return dims[0] if dims else None


def _detect_action_dim(roots: list) -> int | None:
    dims = set(_detect_dims(roots, "action"))
    if not dims:
        return None
    if len(dims) != 1:
        raise ValueError(f"Mixed action dimensions are not supported: {sorted(dims)}")
    return next(iter(dims))


def _detect_state_dim(roots: list) -> int | None:
    dims = _detect_dims(roots, "observation.state")
    return max(dims) if dims else None


def adapt_state_dict_for_state_dim(
    state_dict: dict[str, torch.Tensor],
    saved_state_dim: int | None,
    current_state_dim: int,
) -> dict[str, torch.Tensor]:
    """Adapt a 9D pick checkpoint to the 10D any-contact proprio layout.

    Legacy 9D layout: base6 + target_rel2 + is_grasping.
    New 10D layout:    base6 + target_rel2 + any_contact + is_grasping.
    The old grasp column is split across the two contact columns so the initial
    response is close when both signals are equal, while training can separate
    them.
    """
    if saved_state_dim == current_state_dim:
        return state_dict
    if not (saved_state_dim == 9 and current_state_dim == 10):
        return state_dict

    out = dict(state_dict)
    for key in ("proprio.norm.weight", "proprio.norm.bias"):
        if key in out and out[key].shape[0] == 9:
            old = out[key]
            new_t = old.new_empty(10)
            new_t[:8] = old[:8]
            new_t[8] = old[8]
            new_t[9] = old[8]
            out[key] = new_t
    key = "proprio.proj.weight"
    if key in out and out[key].ndim == 2 and out[key].shape[1] == 9:
        old = out[key]
        new_t = old.new_empty((old.shape[0], 10))
        new_t[:, :8] = old[:, :8]
        new_t[:, 8] = old[:, 8] * 0.5
        new_t[:, 9] = old[:, 8] * 0.5
        out[key] = new_t
    return out


def resolve_frame_caches(roots: list[str | Path], frame_cache: str | Path | None) -> list[str | None]:
    if frame_cache is None:
        return [None] * len(roots)
    cache_text = str(frame_cache)
    caches = cache_text.split(":") if ":" in cache_text else [cache_text]
    if len(roots) == 1:
        if len(caches) != 1:
            raise ValueError("A single --data-root needs exactly one --frame-cache path")
        return caches
    if len(caches) != len(roots):
        raise ValueError(
            "When using multiple --data-root values, --frame-cache must provide "
            "the same number of colon-separated cache paths."
        )
    return caches


def merge_action_stats(stats: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    if not stats:
        raise ValueError("No action stats to merge")
    lows = np.stack([lo for lo, _hi in stats], axis=0)
    highs = np.stack([hi for _lo, hi in stats], axis=0)
    return lows.min(axis=0).astype(np.float32), highs.max(axis=0).astype(np.float32)


def load_merged_action_stats(roots: list[str | Path]) -> tuple[np.ndarray, np.ndarray]:
    from bude_vla.data.action_normalization import load_action_stats

    return merge_action_stats([
        load_action_stats(Path(root) / "meta" / "info.json")
        for root in roots
    ])




def build_dataloader(roots: list[str | Path], chunk_size: int = 4,
                     batch_size: int = 32, num_workers: int = 0,
                     augment: bool = False,
                     n_history_frames: int = 1,
                     lazy_videos: bool = False,
                     lazy_cache_size: int = 8,
                     episode_batches: bool = True,
                     frame_cache: str | Path | None = None,
                     action_stats: tuple[np.ndarray, np.ndarray] | None = None,
                     target_state_dim: int | None = None) -> torch.utils.data.DataLoader:
    all_frames = []
    frame_caches = resolve_frame_caches(roots, frame_cache)
    for root, root_frame_cache in zip(roots, frame_caches):
        ds = BUDETrainingDataset(root, chunk_size=chunk_size, augment=augment,
                                 n_history_frames=n_history_frames,
                                 lazy_videos=lazy_videos,
                                 lazy_cache_size=lazy_cache_size,
                                 frame_cache=root_frame_cache,
                                 action_stats=action_stats,
                                 target_state_dim=target_state_dim)
        ds.read()
        all_frames.append(ds)
        cache_note = f" cache={root_frame_cache}" if root_frame_cache is not None else ""
        print(f"  loaded {len(ds)} frames from {root}{cache_note}")

    if len(all_frames) == 1:
        combined = all_frames[0]
    else:
        combined = torch.utils.data.ConcatDataset(all_frames)

    if frame_cache is not None:
        print("  frame cache: random-access cached frames enabled")

    use_episode_batches = (
        lazy_videos
        and frame_cache is None
        and episode_batches
        and len(all_frames) == 1
        and isinstance(combined, BUDETrainingDataset)
    )
    if use_episode_batches:
        sampler = PhaseBalancedBatchSampler(combined, batch_size=batch_size)
        dl = torch.utils.data.DataLoader(
            combined, batch_sampler=sampler,
            num_workers=num_workers, collate_fn=collate_fn,
            pin_memory=False, persistent_workers=num_workers > 0,
        )
        print("  lazy video batching: phase-balanced multi-episode batches enabled")
    else:
        dl = torch.utils.data.DataLoader(
            combined, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate_fn,
            pin_memory=False, drop_last=True,
            persistent_workers=num_workers > 0,
        )
    print(f"  total frames: {len(combined)}, batches/epoch: {len(dl)}, "
          f"history_frames: {n_history_frames}")
    return dl


def run_closed_loop_eval(policy, cfg, action_lo, action_hi, device,
                          num_episodes: int = 5,
                          max_steps_per_try: int = 300,
                          max_tries: int = 1,
                          seed: int = 123,
                          grid_size: int = 0,
                          eval_state: dict | None = None) -> tuple[float, int, int]:
    """Run closed-loop pick rollouts with current policy weights.

    Returns (success_rate, n_success, n_episodes). Caches MuJoCo model +
    PolicyRolloutRunner in eval_state across calls to avoid repeated EGL
    context creation.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from bude_vla.env_runner import PolicyRolloutRunner
    from bude_vla.envs.so101_mjx import (
        PICK_WORKSPACE_X_RANGE,
        PICK_WORKSPACE_Y_RANGE,
        POLICY_CONTROL_SUBSTEPS,
        load_arm_model,
    )

    was_training = policy.training
    policy.eval()

    if eval_state is None:
        eval_state = {}

    if "model" not in eval_state:
        eval_state["model"] = load_arm_model()
        eval_state["data"] = mujoco.MjData(eval_state["model"])
        eval_state["runner"] = PolicyRolloutRunner(
            eval_state["model"],
            img_size=cfg.img_size,
            max_steps_per_try=max_steps_per_try,
            max_tries=max_tries,
            device=device,
            action_lo=action_lo,
            action_hi=action_hi,
            n_history_frames=cfg.n_history_frames,
            state_dim=cfg.state_dim,
            ensembling=False,
            arm_smooth_steps=POLICY_CONTROL_SUBSTEPS,
            arm_step_frac=1.0,
        )

    runner = eval_state["runner"]
    data = eval_state["data"]

    if grid_size > 1:
        xs = np.linspace(*PICK_WORKSPACE_X_RANGE, grid_size)
        ys = np.linspace(*PICK_WORKSPACE_Y_RANGE, grid_size)
        cube_positions = [np.array([x, y]) for y in ys for x in xs]
    else:
        rng = np.random.default_rng(seed)
        cube_positions = [
            np.array([
                float(rng.uniform(*PICK_WORKSPACE_X_RANGE)),
                float(rng.uniform(*PICK_WORKSPACE_Y_RANGE)),
            ])
            for _ in range(num_episodes)
        ]

    n_success = 0
    for cube_xy in cube_positions:
        result = runner.run_one(data, policy, cube_xy)
        n_success += int(result.success)

    if was_training:
        policy.train()

    n_total = len(cube_positions)
    rate = n_success / n_total if n_total > 0 else 0.0
    return rate, n_success, n_total


def train(
    data_roots: list[str] | None = None,
    ckpt_dir: str = "/home/aditya/bude_vla/checkpoints",
    video_dir: str = "/home/aditya/bude_vla/demos/videos",
    n_steps: int = 50000,
    batch_size: int = 8,
    grad_accum_steps: int = 1,
    chunk_size: int = 16,
    img_size: int = 224,
    lr: float = 3e-4,
    backbone_lr: float = 1e-5,
    weight_decay: float = 1e-4,
    save_every: int = 5000,
    device: str = "cuda",
    augment: bool = False,
    task_name: str = "policy",
    resume: str | None = None,
    init_from: str | None = None,
    init_from_raw: bool = False,
    num_workers: int = 0,
    use_dinov2: bool = False,
    dinov2_finetune_blocks: int = 4,
    use_minilm: bool = False,
    n_history_frames: int = 1,
    eval_every: int = 0,
    eval_episodes: int = 5,
    eval_max_steps: int = 300,
    eval_max_tries: int = 1,
    eval_seed: int = 123,
    eval_grid_size: int = 0,
    train_seed: int = 0,
    ema_decay: float = 0.999,
    use_bc_head: bool = True,
    bc_loss_weight: float = 1.0,
    flow_loss_weight: float = 1.0,
    use_visual_action_cond: bool = True,
    use_context_action_head: bool = True,
    use_perception: bool = True,
    use_perception_action_cond: bool = True,
    early_bc_weight: float = 12.0,
    early_bc_frac: float = 0.22,
    late_bc_weight: float = 1.0,
    late_bc_frac: float = 0.35,
    gripper_loss_weight: float = 1.0,
    shoulder_pan_loss_weight: float = 1.0,
    shoulder_lift_loss_weight: float = 1.0,
    use_gripper_trigger_head: bool = False,
    gripper_trigger_loss_weight: float = 1.0,
    gripper_trigger_label_threshold: float = 0.0,
    gripper_trigger_threshold: float = 0.5,
    gripper_trigger_close_value: float = -1.0,
    action_space: str = "joint_abs",
    ee_delta_scale: float = 0.05,
    allow_action_head_mismatch: bool = False,
    lazy_videos: bool = False,
    lazy_cache_size: int = 8,
    episode_batches: bool = True,
    frame_cache: str | None = None,
):
    if data_roots is None:
        raise ValueError(
            "Must specify --data-root. Default reach+push data is not "
            "appropriate for pick-and-place training."
        )
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)

    ckpt_dir = Path(ckpt_dir) / task_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    cfg = BUDEConfig()
    cfg.img_size = img_size
    cfg.patch_size = 16
    cfg.chunk_size = chunk_size
    cfg.use_dinov2 = use_dinov2
    cfg.dinov2_finetune_blocks = dinov2_finetune_blocks
    cfg.use_minilm = use_minilm
    cfg.n_history_frames = n_history_frames
    cfg.use_bc_head = use_bc_head
    cfg.use_visual_action_cond = use_visual_action_cond
    cfg.use_context_action_head = use_context_action_head
    cfg.use_perception = use_perception
    cfg.use_perception_action_cond = use_perception_action_cond
    cfg.perception_dim = 3
    cfg.use_gripper_trigger_head = use_gripper_trigger_head
    cfg.gripper_trigger_threshold = gripper_trigger_threshold
    cfg.gripper_trigger_close_value = gripper_trigger_close_value
    cfg.action_space = action_space
    cfg.ee_delta_scale = ee_delta_scale
    # Auto-detect action/state dims from dataset if possible; fall back to 6 (SO-101 5-arm + 1-grip).
    cfg.action_dim = action_dim_override if (action_dim_override := _detect_action_dim(data_roots)) else 6
    cfg.state_dim  = state_dim_override  if (state_dim_override  := _detect_state_dim(data_roots))  else 6

    policy = BUDEPolicy(cfg).to(device)

    if init_from is not None and resume is not None:
        raise ValueError("Use either --init-from for weight initialization or --resume for exact continuation, not both.")
    if init_from_raw and init_from is None:
        raise ValueError("--init-from-raw requires --init-from.")

    if init_from is not None:
        init_ckpt = torch.load(init_from, map_location=device, weights_only=False)
        saved_cfg = init_ckpt.get("config", {})
        _checks = [
            ("img_size", cfg.img_size, saved_cfg.get("img_size")),
            ("chunk_size", cfg.chunk_size, saved_cfg.get("chunk_size")),
            ("n_history_frames", cfg.n_history_frames, saved_cfg.get("n_history_frames")),
            ("use_dinov2", cfg.use_dinov2, saved_cfg.get("use_dinov2")),
            ("dinov2_finetune_blocks", cfg.dinov2_finetune_blocks, saved_cfg.get("dinov2_finetune_blocks")),
            ("use_minilm", cfg.use_minilm, saved_cfg.get("use_minilm")),
        ]
        if not allow_action_head_mismatch:
            _checks.append(("action_dim", cfg.action_dim, saved_cfg.get("action_dim")))
            _checks.append(("action_space", cfg.action_space, saved_cfg.get("action_space", "joint_abs")))
        saved_state_dim = saved_cfg.get("state_dim")
        if saved_state_dim is not None and saved_state_dim != cfg.state_dim:
            if not (saved_state_dim == 9 and cfg.state_dim == 10):
                _checks.append(("state_dim", cfg.state_dim, saved_state_dim))
        _mismatches = [f"{name}: checkpoint={saved!r} vs current CLI={cur!r}"
                       for name, cur, saved in _checks
                       if saved is not None and saved != cur]
        if _mismatches:
            raise ValueError(
                "Refusing --init-from: architecture/data flags do not match.\n  "
                + "\n  ".join(_mismatches)
            )
        if init_from_raw:
            init_sd = init_ckpt["model_state_dict"]
            print("  using raw model_state_dict for initialization")
        else:
            init_sd = init_ckpt.get("ema_state_dict") or init_ckpt["model_state_dict"]
        init_sd = adapt_state_dict_for_state_dim(init_sd, saved_cfg.get("state_dim"), cfg.state_dim)
        if allow_action_head_mismatch:
            current_sd = policy.state_dict()
            dropped = []
            filtered = {}
            for k, v in init_sd.items():
                if k in current_sd and tuple(current_sd[k].shape) != tuple(v.shape):
                    dropped.append(k)
                else:
                    filtered[k] = v
            init_sd = filtered
            if dropped:
                print(f"  dropped {len(dropped)} shape-mismatched tensors for new action head/action_dim")
        missing, unexpected = policy.load_state_dict(init_sd, strict=False)
        allowed_prefixes = ("action_head.", "bc_action_head.", "action_cond_proj.", "context_action_head.", "perception_proj.", "gripper_trigger_head.")
        allowed_missing = [k for k in missing if k.startswith(allowed_prefixes)]
        other_missing = [k for k in missing if not k.startswith(allowed_prefixes)]
        if other_missing or unexpected:
            raise ValueError(
                "Unexpected --init-from state_dict mismatch:\n"
                f"  missing={other_missing}\n  unexpected={unexpected}"
            )
        print(f"  initialized weights from {init_from}")
        if allowed_missing:
            print(f"  initialized new modules from scratch ({len(allowed_missing)} tensors)")

    n_params = policy.n_params()
    print(f"Model parameters: {n_params['total']:,}")
    for k, v in n_params.items():
        print(f"  {k}: {v:,}")

    # ── EMA weights ────────────────────────────────────────────────────
    # Flow-matching/diffusion-style policies are notoriously noisy step to
    # step; the raw weights at any given checkpoint can be a much worse
    # policy than the recent average. EMA weights are the de-facto standard
    # fix (used by basically every diffusion policy / DDPM-adjacent paper)
    # and cost ~nothing. eval/final checkpoints prefer EMA weights when
    # ema_decay > 0.
    ema_enabled = ema_decay > 0.0
    ema_state = {k: v.detach().clone() for k, v in policy.state_dict().items()} if ema_enabled else None
    if ema_enabled:
        print(f"  EMA enabled: decay={ema_decay}")

    _action_lo, _action_hi = load_merged_action_stats(data_roots)
    print(f"  action_norm lo={_action_lo[:3]}  hi={_action_hi[:3]}")

    dl = build_dataloader(data_roots, chunk_size=chunk_size,
                          batch_size=batch_size, num_workers=num_workers,
                          augment=augment,
                          n_history_frames=n_history_frames,
                          lazy_videos=lazy_videos,
                          lazy_cache_size=lazy_cache_size,
                          episode_batches=episode_batches,
                          frame_cache=frame_cache,
                          action_stats=(_action_lo, _action_hi),
                          target_state_dim=cfg.state_dim)

    # Differential LR: pretrained DINOv2 backbone gets backbone_lr,
    # new modules (proj, text, proprio, backbone transformer, action head) get lr.
    # NOTE: patch_embed is excluded from pretrained group — with n_history_frames>1,
    # most input channels are zero-initialized and need the full lr, not backbone_lr.
    pretrained_params = []
    new_params = []
    for name, p in policy.named_parameters():
        if not p.requires_grad:
            continue
        if use_dinov2 and "vision.backbone" in name and "patch_embed" not in name:
            pretrained_params.append(p)
        else:
            new_params.append(p)
    param_groups = [
        {"params": pretrained_params, "lr": backbone_lr},
        {"params": new_params, "lr": lr},
    ]
    optimizer = AdamW(param_groups, weight_decay=weight_decay)
    print(f"  pretrained backbone params: {sum(p.numel() for p in pretrained_params):,} (lr={backbone_lr})")
    print(f"  new module params: {sum(p.numel() for p in new_params):,} (lr={lr})")
    print(f"  effective batch size: {batch_size * grad_accum_steps}")

    scaler = GradScaler("cuda", enabled=(device == "cuda"))

    # Scheduler in optimizer-update units (not microbatch units).
    # With grad_accum_steps > 1, scheduler.step() fires once per accumulation
    # cycle, so the schedule horizon must match optimizer updates, not microbatches.
    total_opt_steps = max(1, n_steps // grad_accum_steps)
    warmup_opt_steps = max(1, 2000 // grad_accum_steps)
    def lr_lambda(opt_step):
        if opt_step < warmup_opt_steps:
            return opt_step / warmup_opt_steps
        progress = (opt_step - warmup_opt_steps) / max(1, total_opt_steps - warmup_opt_steps)
        progress = min(progress, 1.0)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = LambdaLR(optimizer, lr_lambda)

    step = 0
    epoch = 0
    running_loss = 0.0
    running_flow_loss = 0.0
    running_bc_loss = 0.0
    running_trigger_loss = 0.0
    grad_norm = torch.tensor(0.0)
    t0 = time.time()
    dl_iter = iter(dl)
    loss_history = []
    eval_history = []
    eval_state: dict = {}

    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=False)

        # Guard against silent corruption: if the CLI flags used for this resume
        # don't match the architecture the checkpoint was actually trained with,
        # policy.load_state_dict can still succeed (shapes may coincidentally
        # match) while optimizer.load_state_dict silently applies Adam momentum
        # to the wrong parameters (param_groups membership shifts when e.g.
        # dinov2_finetune_blocks changes, since only requires_grad=True params
        # are included). This has no crash and no error message — it just makes
        # training quietly worse. Fail loudly instead.
        saved_cfg = ckpt.get("config", {})
        _checks = [
            ("img_size", cfg.img_size, saved_cfg.get("img_size")),
            ("chunk_size", cfg.chunk_size, saved_cfg.get("chunk_size")),
            ("n_history_frames", cfg.n_history_frames, saved_cfg.get("n_history_frames")),
            ("use_dinov2", cfg.use_dinov2, saved_cfg.get("use_dinov2")),
            ("dinov2_finetune_blocks", cfg.dinov2_finetune_blocks, saved_cfg.get("dinov2_finetune_blocks")),
            ("use_minilm", cfg.use_minilm, saved_cfg.get("use_minilm")),
            ("action_dim", cfg.action_dim, saved_cfg.get("action_dim")),
            ("state_dim", cfg.state_dim, saved_cfg.get("state_dim")),
            ("use_bc_head", cfg.use_bc_head, saved_cfg.get("use_bc_head", False)),
            ("use_visual_action_cond", cfg.use_visual_action_cond, saved_cfg.get("use_visual_action_cond", False)),
            ("use_context_action_head", cfg.use_context_action_head, saved_cfg.get("use_context_action_head", False)),
            ("use_perception", cfg.use_perception, saved_cfg.get("use_perception", False)),
            ("use_perception_action_cond", cfg.use_perception_action_cond, saved_cfg.get("use_perception_action_cond", False)),
            ("use_gripper_trigger_head", cfg.use_gripper_trigger_head, saved_cfg.get("use_gripper_trigger_head", False)),
            ("action_space", cfg.action_space, saved_cfg.get("action_space", "joint_abs")),
        ]
        _mismatches = [f"{name}: checkpoint={saved!r} vs current CLI={cur!r}"
                       for name, cur, saved in _checks
                       if saved is not None and saved != cur]
        if _mismatches:
            raise ValueError(
                "Refusing to resume: CLI flags don't match the checkpoint's "
                "saved training config. Continuing anyway risks silently "
                "corrupting the optimizer state (wrong Adam momentum applied "
                "to the wrong parameters) with no crash and no warning.\n  "
                + "\n  ".join(_mismatches)
                + "\nEither fix your CLI flags to match, or if this change is "
                  "intentional, delete the optimizer_state_dict from the "
                  "checkpoint before resuming (this restarts Adam's momentum "
                  "but keeps the trained weights)."
            )

        policy.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        step = ckpt["step"]
        loss_history = ckpt.get("loss_history", [])
        eval_history = ckpt.get("eval_history", [])
        if ema_enabled:
            if "ema_state_dict" in ckpt:
                ema_state = {k: v.to(device) for k, v in ckpt["ema_state_dict"].items()}
                print("  resumed EMA weights from checkpoint")
            else:
                # Older checkpoint with no EMA tracking: reseed from the
                # resumed raw weights rather than silently keeping the EMA
                # snapshot from before load_state_dict (which would be from
                # a freshly-initialized, untrained policy).
                ema_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
                print("  no EMA state in checkpoint; reseeded EMA from resumed raw weights")
        # Fast-forward scheduler using optimizer-update units (not microbatch)
        opt_steps_done = step // grad_accum_steps
        for _ in range(opt_steps_done):
            scheduler.step()
        running_loss = 0.0
        dl_iter = iter(dl)
        print(f"  resumed from step {step}, lr={scheduler.get_last_lr()[0]:.2e}, "
              f"loss_hist entries={len(loss_history)}")

    best_eval_rate = max(
        (float(rate) for _eval_step, rate in eval_history), default=-1.0
    )

    while step < n_steps:
        if step % grad_accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        try:
            batch = next(dl_iter)
        except StopIteration:
            epoch += 1
            dl_iter = iter(dl)
            batch = next(dl_iter)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with autocast("cuda", enabled=(device == "cuda")):
            out = policy(batch)
            v_pred = out["velocity"]
            v_target = batch["actions"] - batch["noise"]
            mask = batch["mask"].unsqueeze(-1)  # (B, chunk_size, 1)
            flow_dim_w = build_action_dim_weights(
                v_pred.shape[-1], mask.device, mask.dtype, gripper_loss_weight,
                shoulder_pan_loss_weight, shoulder_lift_loss_weight)
            flow_dim_w_view = flow_dim_w.view(1, 1, -1)
            flow_denom = (mask * flow_dim_w_view).sum().clamp_min(1.0)
            flow_loss = (((v_pred - v_target) ** 2) * mask * flow_dim_w_view).sum() / flow_denom
            if "bc_actions" in out:
                bc_err = ((out["bc_actions"] - batch["actions"]) ** 2) * mask
                sample_w, dim_w, bc_denom = build_bc_loss_weights(
                    phase=batch.get("phase"),
                    mask=mask.squeeze(-1),
                    action_dim=v_pred.shape[-1],
                    early_bc_frac=early_bc_frac,
                    early_bc_weight=early_bc_weight,
                    late_bc_frac=late_bc_frac,
                    late_bc_weight=late_bc_weight,
                    gripper_loss_weight=gripper_loss_weight,
                    shoulder_pan_loss_weight=shoulder_pan_loss_weight,
                    shoulder_lift_loss_weight=shoulder_lift_loss_weight,
                )
                bc_loss = (bc_err * sample_w * dim_w.view(1, 1, -1)).sum() / bc_denom
                loss = flow_loss_weight * flow_loss + bc_loss_weight * bc_loss
            else:
                bc_loss = torch.zeros((), device=device)
                loss = flow_loss
            if "gripper_close_logits" in out:
                close_target = (batch["actions"][..., -1] <= gripper_trigger_label_threshold).to(out["gripper_close_logits"].dtype)
                trigger_mask = mask.squeeze(-1)
                trigger_raw = nn.functional.binary_cross_entropy_with_logits(
                    out["gripper_close_logits"], close_target, reduction="none")
                trigger_loss = (trigger_raw * trigger_mask).sum() / trigger_mask.sum().clamp_min(1.0)
                loss = loss + gripper_trigger_loss_weight * trigger_loss
            else:
                trigger_loss = torch.zeros((), device=device)
            loss_scaled = loss / grad_accum_steps

        scaler.scale(loss_scaled).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_ran = scaler.get_scale() >= scale_before
            if optimizer_ran:
                scheduler.step()
                if ema_enabled:
                    with torch.no_grad():
                        for k, v in policy.state_dict().items():
                            if torch.is_floating_point(v):
                                ema_state[k].mul_(ema_decay).add_(v, alpha=1.0 - ema_decay)
                            else:
                                ema_state[k].copy_(v)  # non-float buffers: copy

        running_loss += loss.item()
        running_flow_loss += flow_loss.item()
        running_bc_loss += bc_loss.item()
        running_trigger_loss += trigger_loss.item()
        step += 1

        if step % 100 == 0:
            avg = running_loss / 100
            flow_avg = running_flow_loss / 100
            bc_avg = running_bc_loss / 100
            trigger_avg = running_trigger_loss / 100
            elapsed = time.time() - t0
            sps = step / elapsed
            loss_history.append((step, avg))
            print(f"step {step:6d} | loss {avg:.6f} | flow {flow_avg:.6f} | "
                  f"bc {bc_avg:.6f} | grip_cls {trigger_avg:.6f} | grad_norm {float(grad_norm):.3f} | "
                  f"lr {scheduler.get_last_lr()[0]:.2e} | "
                  f"{sps:.1f} steps/s | epoch {epoch}")
            running_loss = 0.0
            running_flow_loss = 0.0
            running_bc_loss = 0.0
            running_trigger_loss = 0.0

        if step % save_every == 0:
            ckpt_path = ckpt_dir / f"{task_name}_step_{step:06d}.pt"
            torch.save({
                "step": step,
                "model_state_dict": policy.state_dict(),
                "ema_state_dict": ema_state if ema_enabled else None,
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "loss_history": loss_history,
                "eval_history": eval_history,
                "action_norm_lo": _action_lo.tolist(),
                "action_norm_hi": _action_hi.tolist(),
                "config": {
                    "use_dinov2": cfg.use_dinov2,
                    "dinov2_finetune_blocks": cfg.dinov2_finetune_blocks,
                    "use_minilm": cfg.use_minilm,
                    "n_history_frames": cfg.n_history_frames,
                    "img_size": cfg.img_size,
                    "chunk_size": cfg.chunk_size,
                    "action_dim": cfg.action_dim,
                    "state_dim": cfg.state_dim,
                    "ema_decay": ema_decay if ema_enabled else None,
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "use_bc_head": cfg.use_bc_head,
                    "bc_loss_weight": bc_loss_weight if cfg.use_bc_head else None,
                    "flow_loss_weight": flow_loss_weight,
                    "early_bc_weight": early_bc_weight if cfg.use_bc_head else None,
                    "early_bc_frac": early_bc_frac if cfg.use_bc_head else None,
                    "late_bc_weight": late_bc_weight if cfg.use_bc_head else None,
                    "late_bc_frac": late_bc_frac if cfg.use_bc_head else None,
                    "gripper_loss_weight": gripper_loss_weight if cfg.use_bc_head else None,
                    "shoulder_pan_loss_weight": shoulder_pan_loss_weight if cfg.use_bc_head else None,
                    "shoulder_lift_loss_weight": shoulder_lift_loss_weight if cfg.use_bc_head else None,
                    "use_visual_action_cond": cfg.use_visual_action_cond,
                    "use_context_action_head": cfg.use_context_action_head,
                    "use_perception": cfg.use_perception,
                    "use_perception_action_cond": cfg.use_perception_action_cond,
                    "perception_dim": cfg.perception_dim,
                    "use_gripper_trigger_head": cfg.use_gripper_trigger_head,
                    "gripper_trigger_loss_weight": gripper_trigger_loss_weight if cfg.use_gripper_trigger_head else None,
                    "gripper_trigger_label_threshold": gripper_trigger_label_threshold if cfg.use_gripper_trigger_head else None,
                    "gripper_trigger_threshold": cfg.gripper_trigger_threshold if cfg.use_gripper_trigger_head else None,
                    "gripper_trigger_close_value": cfg.gripper_trigger_close_value if cfg.use_gripper_trigger_head else None,
                    "action_space": cfg.action_space,
                    "ee_delta_scale": cfg.ee_delta_scale,
                },
            }, ckpt_path)
            print(f"  saved checkpoint: {ckpt_path}")

        if eval_every > 0 and step > 0 and step % eval_every == 0:
            if ema_enabled:
                # EMA weights are what actually gets deployed at inference
                # time, so measure success against those, not the noisier
                # raw weights. Swap in, eval, swap back.
                _raw_backup = {k: v.detach().clone() for k, v in policy.state_dict().items()}
                policy.load_state_dict(ema_state)
            eval_rate, eval_n_success, eval_n_total = run_closed_loop_eval(
                policy, cfg, _action_lo, _action_hi, device,
                num_episodes=eval_episodes,
                max_steps_per_try=eval_max_steps,
                max_tries=eval_max_tries,
                seed=eval_seed,
                grid_size=eval_grid_size,
                eval_state=eval_state,
            )
            if ema_enabled:
                policy.load_state_dict(_raw_backup)
            eval_history.append((step, eval_rate))
            print(f"  [eval{'/ema' if ema_enabled else ''}] step {step:6d} | closed-loop success "
                  f"{eval_n_success}/{eval_n_total} ({eval_rate*100:.0f}%)")
            if eval_rate >= best_eval_rate:
                best_eval_rate = eval_rate
                source = ckpt_dir / f"{task_name}_step_{step:06d}.pt"
                best = ckpt_dir / f"{task_name}_best.pt"
                if source.exists():
                    shutil.copy2(source, best)
                    print(
                        f"  new best checkpoint: {best} "
                        f"(closed-loop success {eval_rate * 100:.0f}%)"
                    )
                else:
                    print("  warning: eval improved but matching step checkpoint is missing")

    # Flush any remaining accumulated gradients
    if step % grad_accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_ran = scaler.get_scale() >= scale_before
        if optimizer_ran:
            scheduler.step()
            if ema_enabled:
                with torch.no_grad():
                    for k, v in policy.state_dict().items():
                        if torch.is_floating_point(v):
                            ema_state[k].mul_(ema_decay).add_(v, alpha=1.0 - ema_decay)
                        else:
                            ema_state[k].copy_(v)

    final_ckpt = ckpt_dir / f"{task_name}_final.pt"
    torch.save({
        "step": step,
        "model_state_dict": policy.state_dict(),
        "ema_state_dict": ema_state if ema_enabled else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss_history": loss_history,
        "eval_history": eval_history,
        "action_norm_lo": _action_lo.tolist(),
        "action_norm_hi": _action_hi.tolist(),
        "config": {
            "use_dinov2": cfg.use_dinov2,
            "dinov2_finetune_blocks": cfg.dinov2_finetune_blocks,
            "use_minilm": cfg.use_minilm,
            "n_history_frames": cfg.n_history_frames,
            "img_size": cfg.img_size,
            "chunk_size": cfg.chunk_size,
            "action_dim": cfg.action_dim,
            "state_dim": cfg.state_dim,
            "ema_decay": ema_decay if ema_enabled else None,
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "use_bc_head": cfg.use_bc_head,
            "bc_loss_weight": bc_loss_weight if cfg.use_bc_head else None,
            "flow_loss_weight": flow_loss_weight,
            "early_bc_weight": early_bc_weight if cfg.use_bc_head else None,
            "early_bc_frac": early_bc_frac if cfg.use_bc_head else None,
            "late_bc_weight": late_bc_weight if cfg.use_bc_head else None,
            "late_bc_frac": late_bc_frac if cfg.use_bc_head else None,
            "gripper_loss_weight": gripper_loss_weight if cfg.use_bc_head else None,
            "shoulder_pan_loss_weight": shoulder_pan_loss_weight if cfg.use_bc_head else None,
            "shoulder_lift_loss_weight": shoulder_lift_loss_weight if cfg.use_bc_head else None,
            "use_visual_action_cond": cfg.use_visual_action_cond,
            "use_context_action_head": cfg.use_context_action_head,
            "use_perception": cfg.use_perception,
            "use_perception_action_cond": cfg.use_perception_action_cond,
            "perception_dim": cfg.perception_dim,
            "use_gripper_trigger_head": cfg.use_gripper_trigger_head,
            "gripper_trigger_loss_weight": gripper_trigger_loss_weight if cfg.use_gripper_trigger_head else None,
            "gripper_trigger_label_threshold": gripper_trigger_label_threshold if cfg.use_gripper_trigger_head else None,
            "gripper_trigger_threshold": cfg.gripper_trigger_threshold if cfg.use_gripper_trigger_head else None,
            "gripper_trigger_close_value": cfg.gripper_trigger_close_value if cfg.use_gripper_trigger_head else None,
            "action_space": cfg.action_space,
            "ee_delta_scale": cfg.ee_delta_scale,
        },
    }, final_ckpt)
    print(f"Training done in {time.time()-t0:.0f}s. Final checkpoint: {final_ckpt}")

    if "runner" in eval_state:
        eval_state["runner"].close()

    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps. Effective batch = "
                             "batch_size * grad_accum_steps. Use 4 with "
                             "--batch-size 8 to simulate batch 32 on 8GB GPU.")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224,
                        help="Image resolution for ViT input.")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate for new modules (backbone, action head, etc.)")
    parser.add_argument("--backbone-lr", type=float, default=1e-5,
                        help="Learning rate for pretrained DINOv2 backbone. "
                             "Lower than --lr to preserve pretrained features.")
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--data-root", action="append", default=None,
                        help="Dataset root(s) to train on. May be passed "
                             "multiple times. Required — no default.")
    parser.add_argument("--task", default="policy",
                        help="Subdirectory under checkpoints/ for this run, "
                             "and filename prefix (e.g. 'pick', 'reach').")
    parser.add_argument("--augment", action="store_true",
                        help="Enable image augmentation (random crop + "
                             "brightness jitter) on the training dataset.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0=single-thread). Use 2-4 "
                             "to keep GPU fed when augmentation is on.")
    parser.add_argument("--lazy-videos", action="store_true", default=False,
                        help="Decode MP4s lazily per-episode instead of "
                             "predecoding all_images.npy. Use when the .npy "
                             "does not fit on disk or is corrupted.")
    parser.add_argument("--lazy-cache-size", type=int, default=1,
                        help="Per-worker LRU cache size (num episodes held decoded in RAM). Use 1 with lazy episode batches to avoid RAM spikes.")
    parser.add_argument("--no-episode-batches", action="store_true",
                        help="Disable phase-balanced lazy-video batch sampling. Normal random frame shuffling is much more RAM/CPU intensive.")
    parser.add_argument("--frame-cache", default=None,
                        help="Path to a prebuilt stacked frame cache from scripts/build_frame_cache.py. Avoids MP4 decoding during training.")
    parser.add_argument("--no-bc-head", action="store_true",
                        help="Disable the deterministic behavior-cloning action head. New training enables it by default for stable receding-horizon rollout.")
    parser.add_argument("--bc-loss-weight", type=float, default=1.0,
                        help="Weight for direct normalized action-chunk MSE when the BC head is enabled.")
    parser.add_argument("--flow-loss-weight", type=float, default=1.0,
                        help="Weight for auxiliary flow-matching loss. Lower this when deploying the BC/context head.")
    parser.add_argument("--no-visual-action-cond", action="store_true",
                        help="Do not add pooled visual patch tokens to the action conditioning vector.")
    parser.add_argument("--no-context-action-head", action="store_true",
                        help="Disable the spatial context action decoder and use the older vector BC head.")
    parser.add_argument("--no-perception", action="store_true",
                        help="Disable the pixel-derived red-cube centroid token.")
    parser.add_argument("--no-perception-action-cond", action="store_true",
                        help="Disable direct pixel-centroid conditioning in the action decoder.")
    parser.add_argument("--early-bc-weight", type=float, default=12.0,
                        help="Extra BC loss weight for early approach frames where cube visual grounding matters most.")
    parser.add_argument("--early-bc-frac", type=float, default=0.22,
                        help="Episode phase threshold for early-frame BC weighting.")
    parser.add_argument("--late-bc-weight", type=float, default=1.0,
                        help="Extra BC loss weight for late close/lift/move frames.")
    parser.add_argument("--late-bc-frac", type=float, default=0.35,
                        help="Episode phase threshold where late-frame BC weighting starts.")
    parser.add_argument("--gripper-loss-weight", type=float, default=1.0,
                        help="BC loss multiplier for the gripper action dimension.")
    parser.add_argument("--shoulder-pan-loss-weight", type=float, default=1.0,
                        help="Flow and BC loss multiplier for shoulder-pan precision.")
    parser.add_argument("--shoulder-lift-loss-weight", type=float, default=1.0,
                        help="Flow and BC loss multiplier for shoulder-lift precision.")
    parser.add_argument("--use-gripper-trigger-head", action="store_true",
                        help="Add a binary BCE head for sharp close-now gripper decisions.")
    parser.add_argument("--gripper-trigger-loss-weight", type=float, default=1.0,
                        help="Loss weight for the binary gripper close trigger head.")
    parser.add_argument("--gripper-trigger-label-threshold", type=float, default=0.0,
                        help="Normalized action threshold: action gripper <= threshold means close label.")
    parser.add_argument("--gripper-trigger-threshold", type=float, default=0.5,
                        help="Sigmoid probability threshold for forcing close at inference.")
    parser.add_argument("--gripper-trigger-close-value", type=float, default=-1.0,
                        help="Normalized gripper command used when trigger predicts close.")
    parser.add_argument("--action-space", choices=["joint_abs", "ee_delta", "ee_abs"], default="joint_abs",
                        help="Action representation. ee_delta predicts TCP offsets; ee_abs predicts absolute TCP targets. Both execute through IK.")
    parser.add_argument("--ee-delta-scale", type=float, default=0.05,
                        help="Max expected TCP delta magnitude in meters for ee_delta rollouts.")
    parser.add_argument("--allow-action-head-mismatch", action="store_true",
                        help="Allow --init-from when action_dim/action heads changed; transfers shared trunk weights and reinitializes mismatched action heads.")
    parser.add_argument("--init-from", default=None,
                        help="Initialize model weights from a checkpoint without loading optimizer/scheduler state. Useful for adding the BC head to an existing flow checkpoint.")
    parser.add_argument("--init-from-raw", action="store_true",
                        help="With --init-from, initialize from model_state_dict instead of EMA weights.")
    parser.add_argument("--resume", default=None,
                        help="Path to a checkpoint to resume training from. "
                             "Restores model, optimizer, step counter, and "
                             "loss history; continues until --n-steps.")
    parser.add_argument("--finetune-blocks", type=int, default=4,
                        help="Number of DINOv2 blocks to fine-tune (1 = 12 total; "
                             "4 default, means only last 4 blocks trainable. "
                             "Full fine-tuning at backbone_lr. Use 0 to unfreeze all.")
    parser.add_argument("--use-dinov2", action="store_true",                        help="Replace from-scratch ViT with frozen pretrained "
                             "DINOv2-small backbone (P0 architecture review fix).")
    parser.add_argument("--use-minilm", action="store_true",
                        help="Replace TinyTextEncoder with frozen pretrained "
                             "MiniLM (sentence-transformers/all-MiniLM-L6-v2).")
    parser.add_argument("--n-history-frames", type=int, default=1,
                        help="Number of stacked history frames per observation. "
                             "1=Markovian (default), 2=allows velocity inference.")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Run closed-loop pick-and-place eval every N steps "
                             "(0 = disabled). Recommended: same as --save-every.")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Cube positions per closed-loop eval pass.")
    parser.add_argument("--eval-max-steps", type=int, default=300,
                        help="Max env steps per eval episode (shorter than "
                             "the 4000 used in eval_pick_ball.py — cheap "
                             "progress signal, not a final benchmark).")
    parser.add_argument("--eval-max-tries", type=int, default=1,
                        help="Retries per eval episode. 1 = single-shot.")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay for a shadow copy of the weights (0 disables). "
                             "EMA weights are used for closed-loop eval when enabled; "
                             "benchmark raw and EMA explicitly before deployment.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Training model and data-order seed.")
    parser.add_argument("--eval-seed", type=int, default=123,
                        help="Fixed seed for random eval positions.")
    parser.add_argument("--eval-grid-size", type=int, default=0,
                        help="If >1, evaluate an NxN deterministic workspace grid instead of random positions.")
    args = parser.parse_args()

    roots = args.data_root
    if roots is None:
        raise ValueError(
            "Must specify --data-root. Old default (reach_v3 + push_v3) "
            "has been removed — those datasets are obsolete for pick training."
        )

    train(
        data_roots=roots,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        chunk_size=args.chunk_size,
        img_size=args.img_size,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        save_every=args.save_every,
        augment=args.augment,
        resume=args.resume,
        init_from=args.init_from,
        init_from_raw=args.init_from_raw,
        task_name=args.task,
        num_workers=args.num_workers,
        use_dinov2=args.use_dinov2,
        dinov2_finetune_blocks=args.finetune_blocks,
        use_minilm=args.use_minilm,
        n_history_frames=args.n_history_frames,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
        eval_max_tries=args.eval_max_tries,
        eval_seed=args.eval_seed,
        eval_grid_size=args.eval_grid_size,
        train_seed=args.seed,
        ema_decay=args.ema_decay,
        use_bc_head=not args.no_bc_head,
        bc_loss_weight=args.bc_loss_weight,
        flow_loss_weight=args.flow_loss_weight,
        use_visual_action_cond=not args.no_visual_action_cond,
        use_context_action_head=not args.no_context_action_head,
        use_perception=not args.no_perception,
        use_perception_action_cond=not args.no_perception_action_cond,
        early_bc_weight=args.early_bc_weight,
        early_bc_frac=args.early_bc_frac,
        late_bc_weight=args.late_bc_weight,
        late_bc_frac=args.late_bc_frac,
        gripper_loss_weight=args.gripper_loss_weight,
        shoulder_pan_loss_weight=args.shoulder_pan_loss_weight,
        shoulder_lift_loss_weight=args.shoulder_lift_loss_weight,
        use_gripper_trigger_head=args.use_gripper_trigger_head,
        gripper_trigger_loss_weight=args.gripper_trigger_loss_weight,
        gripper_trigger_label_threshold=args.gripper_trigger_label_threshold,
        gripper_trigger_threshold=args.gripper_trigger_threshold,
        gripper_trigger_close_value=args.gripper_trigger_close_value,
        action_space=args.action_space,
        ee_delta_scale=args.ee_delta_scale,
        allow_action_head_mismatch=args.allow_action_head_mismatch,
        lazy_videos=args.lazy_videos,
        lazy_cache_size=args.lazy_cache_size,
        episode_batches=not args.no_episode_batches,
        frame_cache=args.frame_cache,
    )
