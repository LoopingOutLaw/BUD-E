# BUD-E VLA — Design Specification

**Date:** 2026-06-13
**Author:** Aditya
**Status:** Draft — pending user review

---

## 1. Project Identity

**Name:** BUD-E (Brave Unified Deployment — Embodied)

**Codename / repo:** `bude_vla`

**One-liner:** A 22M-parameter, soft-prompted, flow-matching Vision-Language-Action model, re-implemented from scratch with reference to the X-VLA architecture, trained on a simulated SO-101 robotic arm in MuJoCo/MJX, and adapted to new tasks via only 1% of params.

**Goals:**
- 100% original code — no copy-pasted lerobot policies or HuggingFace model weights
- Showcase-grade: clean README, animated results, architecture diagrams
- Reference-able: cite X-VLA, SmolVLA, Octo as inspiration
- Publishable: model weights on HuggingFace Hub, repo on GitHub

**License:** Apache-2.0

---

## 2. System Architecture

### 2.1 Overview

BUD-E follows a four-stage pipeline: Vision → Tokenize → Fuse (with soft prompts) → Predict actions via flow-matching.

```
Camera images  ──▶  Vision Tower (ViT-S, 5M)   ──▶ patch_tokens (B, 196, 256)
Text instruction ──▶  Text Encoder (4-layer Xfmr, 3M) ──▶ text_tokens (B, 64, 256)
Joint state (7D) ──▶  Linear Proj (0.05M)      ──▶ state_token (B, 1, 256)
Domain ID       ──▶  Soft Prompts (0.05M)      ──▶ prompts (B, 32, 256)
                                                          │
                                 All concatenated & fed into:
                                                          ▼
                                              Transformer Backbone (8-layer, 12M)
                                                          │
                                                          ▼
                                              Flow-Matching Action Head (2M)
                                                          │
                                                          ▼
                                            velocity field → denoise → action_chunk (B, 32, 7)
```

### 2.2 Component Specifications

#### 2.2.1 Vision Tower — `ViTSmall`

- Patch size: 16×16
- Input: 224×224 RGB
- 12 Transformer layers, dim=384, 6 heads → project to d=256 via 1 linear
- Positional encoding: learnt (not sinusoidal)
- Output: (B, 196, 256)
- **Parameters:** ~5M

#### 2.2.2 Text Encoder — `TinyTextEncoder`

- Custom BPE tokenizer trained on the ~200 unique instruction strings used across all tasks
- Vocab size: 512
- 4-layer Transformer encoder, dim=256, 4 heads
- Max sequence length: 64 tokens
- No pretrained weights — trained end-to-end
- **Parameters:** ~3M

#### 2.2.3 Proprioception Projector

- Single linear layer: (7 → 256) with LayerNorm
- Input: 6 joint angles + 1 gripper state
- **Parameters:** ~0.05M

#### 2.2.4 Transformer Backbone — `PolicyTransformer`

- 8-layer Transformer encoder, d=256, 8 heads, FFN dim=1024
- Input sequence: [soft_prompts(32) | state_token(1) | patch_tokens(196) | text_tokens(≤64)]
- Causal mask on action tokens only; full bidirectional attention on vision+text+prompts
- **Parameters:** ~12M

#### 2.2.5 Flow-Matching Action Head

- Input: noisy action chunk (B, 32, 7), timestep τ, and a query from backbone memory
- 4-layer MLP with sinusoidal time embedding (dim=128), hidden dim=512
- Predicts velocity field v(x_t, τ) where x_t = (1-τ)·x_0 + τ·x_1 (OT-CFM schedule)
- Training: 10 denoising steps, MSE loss on velocity
- Inference: Euler integration over 10 steps
- Training action chunk size: 32 steps
- Inference action chunk size: 8 steps (predict 8, execute first 4, re-observe — temporal ensembling per diffusion-policy)
- **Parameters:** ~2M

#### 2.2.6 Soft Prompts

- Learnable table: (N_domains, N_p=32, d=256)
- N_domains = 6 for Phase I (3 tasks × 2 morphologies), 1 for Phase II
- Prepended to every backbone input before self-attention
- Orthogonal initialization (scaled random normal, then Gram-Schmidt)
- **Parameters per domain:** ~0.008M

### 2.3 Total parameter count

| Component | Params |
|---|---|
| Vision Tower | 5M |
| Text Encoder | 3M |
| Proprio Projector | 0.05M |
| Transformer Backbone | 12M |
| Action Head | 2M |
| Soft Prompts (6 domains) | 0.05M |
| **Total** | **~22M** |

Fits comfortably in 8GB VRAM with batch size 16 in mixed precision (bfloat16 forward, fp32 master weights).

### 2.4 Data shapes (the contract)

```python
# Input
images:    (B, 3, 224, 224)
text_ids:  (B, T_text)          # T_text ≤ 64
proprio:   (B, 7)               # 6 joints + 1 gripper
domain_id: int                  # indexes soft-prompt table

# Internal
patch_tokens: (B, 196, 256)
text_tokens:  (B, T_text, 256)
state_token:  (B, 1, 256)
prompts:      (B, 32, 256)

# Backbone input
tokens: (B, 32+1+196+T_text, 256)  # ≈ 293 tokens

# Output
velocity: (B, 32, 7)            # predicts velocity field
action:   (B, 32, 7)            # after Euler denoising at inference
```

---

## 3. Simulation Environment & Data

### 3.1 Simulator

**MuJoCo + MJX** (GPU-accelerated, JAX-based MuJoCo)

- 256 parallel environments on 4060
- Differentiable physics for future gradient-through-sim experiments
- Direct URDF→MJCF conversion pipeline

### 3.2 Robot

**SO-101** (same as lerobot's SO-101, 5+1 DOF)

- URDF sourced from lerobot GitHub, converted to MJCF via `scripts/urdf_to_mjcf.py`
- Visual meshes replaced with primitive geometries (capsules, cylinders) for GPU speed
- **Morph variant:** forearm link lengthened by 15% — forces soft prompts to differentiate

Action space: 7D — [joint_1…joint_5, gripper] (absolute joint positions)

Observation space:
- RGB image: 224×224 from a fixed over-the-shoulder camera
- Proprioception: 7D joint state
- Language instruction: string

### 3.3 Tasks

| Phase | Task | Description | Heuristic Demo Policy |
|---|---|---|---|
| I | `reach_target` | Move end-effector to a randomly-placed colored sphere | IK + PD control |
| I | `push_cube` | Push a cube to a target zone on the table | IK + vision-servoed offset |
| I | `pick_place_basic` | Pick a cube from random pose, place at a target | Scripted state machine: approach → grasp → lift → place → release |
| I | **morph variant** | Same 3 tasks with the elongated-forearm SO-101 | Same scripted policies (harder, sparser success) |
| II | `pick_place_bowl` | Pick red cube, place in blue bowl on a cluttered tabletop | Scripted pick pipeline with collision avoidance |

### 3.4 Demo generation

- `scripts/generate_demos.py`: runs 256 parallel MJX envs, applies scripted policy, saves episodes
- Output format: **LeRobotDataset v3** (Parquet + MP4) — lingua franca, HF Hub compatible
- Volume: ~100 episodes per task × 7 task variants = 700 episodes, ~50k timesteps
- Generation time: <2 hours on 4060

### 3.5 Dataset splits

- Phase I training: all 700 episodes (6 domains)
- Phase I validation: 10% holdout per task
- Phase II training: 50–100 episodes of `pick_place_bowl` only
- Phase II validation: 10% holdout of `pick_place_bowl`

---

## 4. Training & Evaluation

### 4.1 Phase I — From-scratch pretraining

**Goal:** Learn embodiment-agnostic representations across 6 domains.

- All components trainable
- Multi-task mixture sampling (equal weight per domain)
- 3M total environment steps
- Batch size: 16 trajectories × 32-step chunks
- Optimizer: AdamW
  - Vision + text encoders: lr=1e-5
  - Backbone: lr=1e-4
  - Soft prompts + action head: lr=1e-3
  - Weight decay: 0.01
  - Cosine schedule with 3% warmup
- Mixed precision: bfloat16 forward, fp32 master weights
- Checkpoint every 50k steps
- Logging: W&B (preferred) or TensorBoard
- Estimated wall-clock: 6–10 hours on 4060

### 4.2 Phase II — Soft-prompt adaptation

**Goal:** Adapt to a new task using only ~1% of parameters.

- Load Phase I checkpoint
- Freeze: `vision_encoder.*`, `text_encoder.*`
- Trainable: soft prompts (new table, single domain), last 2 backbone layers, action head
- Trainable param ratio: ~1% of 22M ≈ 220k params
- Duration: ~2000 steps (~10 minutes on 4060)
- Same optimizer settings for trainable components

### 4.3 Evaluation protocol

**Closed-loop eval with temporal ensembling:**
1. Observe (image, proprio, instruction)
2. Predict 8-step action chunk
3. Execute first 4 steps in MJX
4. Re-observe, repeat
5. Continue for max 200 steps or until task success

**50 episodes per task**, fixed seed sweep (seeds 0–49).

**Success criteria:**

| Task | Success definition |
|---|---|
| `reach_target` | EE within 3cm of target for 10 consecutive frames |
| `push_cube` | Cube center within 4cm of target zone for 10 frames |
| `pick_place_basic` | Cube placed within 5cm of target; gripper open; cube not dropped for 10 frames |
| `pick_place_bowl` | Cube inside bowl boundary; released; stable for 10 frames |

### 4.4 Metrics

| Metric | Purpose |
|---|---|
| Task success rate (%) | Primary result — goes in README / HF model card |
| Action MSE on validation set | Training sanity check |
| Soft-prompt cosine similarity matrix (6×6) | Shows domain specialization — key showcase figure |
| Inference latency (ms per action chunk) | 4060 throughput claim for README |
| Trainable parameter count | Validates "1% adaptation" thesis |
| Training loss curve | Standard, for blog post / W&B |

### 4.5 Hardware budget

| Phase | Time | VRAM |
|---|---|---|
| Demo generation | ~1.5 hours | ~3 GB |
| Phase I pretraining | ~6–10 hours | ~6 GB |
| Phase II adaptation | ~10 min | ~4 GB |
| Eval (50 eps × 4 tasks) | ~20 min | ~3 GB |
| **Total** | **~1 workday** | |

---

## 5. Risks, Learning Value & Showcase Plan

### 5.1 Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Flow-matching fails in tiny model (velocity field over-smooths) | Low success rate | Fallback: Gaussian-MSE deterministic head. Switch by changing 1 config flag. |
| Soft prompts collapse to similar vectors across domains | Weak showcase | Orthogonal init + monitor similarity matrix; add more morph variants if needed |
| MJX VRAM blowup during parallel demo-gen | Can't generate enough data | Drop to 128 parallel envs; documented fallback |
| Phase I doesn't converge in 3M steps | Wasted compute | Check val MSE at 1M; if flat, increase to 5M or reduce model dim to 192 |

### 5.2 Learning value

- Deep understanding of **why** X-VLA's architecture looks the way it does (the only real way to internalize a paper)
- Hands-on with **flow matching** — the current SOTA action representation
- **Soft-prompt dynamics** in embodied models — genuinely novel, publishable at workshops
- End-to-end VLA loop: perception → fusion → action, all written by you

### 5.3 Showcase deliverables

1. **GitHub repo** (`bude_vla`) — README with: results table, training curves, soft-prompt similarity heatmap, attention rollout GIFs
2. **Blog post** — "Re-implementing X-VLA in 22M params: what fits on a laptop GPU"
3. **HuggingFace model card** — upload trained weights under your username
4. **Social media GIF** — BUD-E performing pick-place in MJX
5. **Architecture doc** — `/docs/architecture.md` for reproducibility

---

## Appendix: Directory Layout

```
bude_vla/
├── README.md
├── LICENSE                          # Apache-2.0
├── pyproject.toml
├── docs/
│   ├── architecture.md              # detailed architecture writeup
│   └── superpowers/
│       └── specs/
│           └── 2026-06-13-bude-vla-design.md  # this file
├── src/bude_vla/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vision.py               # ViTSmall
│   │   ├── text_encoder.py         # TinyTextEncoder + BPE tokenizer
│   │   ├── proprio.py              # Linear projector
│   │   ├── backbone.py             # 8-layer PolicyTransformer
│   │   ├── action_head.py          # Flow-matching action head
│   │   ├── soft_prompts.py         # Learnable soft-prompt table
│   │   └── policy.py               # Assembles all components
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── so101_mjx.py            # MJX wrapper for SO-101
│   │   └── tasks/                  # Task definitions (reach, push, pick)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py              # LeRobotDataset v3 reader
│   │   └── demo_recorder.py        # Saves scripted demos to disk
│   ├── training/
│   │   ├── __init__.py
│   │   ├── pretrain.py             # Phase I trainer
│   │   ├── adapt.py                # Phase II trainer
│   │   └── lr_schedule.py          # Layerwise LR groups
│   └── deployment/
│       ├── __init__.py
│       └── inference.py            # Closed-loop eval with temporal ensembling
├── scripts/
│   ├── generate_demos.py           # Parallel MJX demo generation
│   ├── train.py                    # CLI entry: Phase I or II
│   ├── eval.py                     # CLI entry: evaluation
│   ├── visualize.py                # Render + GIF export
│   └── urdf_to_mjcf.py             # One-shot URDF→MJCF converter
├── urdf/
│   └── so101.urdf                  # From lerobot (downloaded, committed)
├── assets/
│   └── so101.xml                   # Converted MJCF with primitive meshes
└── configs/
    ├── pretrain_reach.yaml
    ├── pretrain_push.yaml
    ├── pretrain_pick.yaml
    ├── pretrain_morph_reach.yaml
    ├── pretrain_morph_push.yaml
    ├── pretrain_morph_pick.yaml
    └── adapt_pick_place_bowl.yaml
```

## Appendix: Key design decisions log

| Decision | Choice | Rationale |
|---|---|---|
| VLA model | From-scratch 22M re-implementation of X-VLA architecture | Showcase + learning value |
| Robot | SO-101 (from lerobot URDF) | Standardized, well-documented, easy to source |
| Simulator | MuJoCo + MJX | GPU-batchable, differentiable, free, Python-native |
| Action representation | Flow-matching (OT-CFM) | SOTA, matches X-VLA design |
| Adaptation mechanism | Soft prompts | Matches X-VLA thesis, observable specialization |
| Dataset format | LeRobotDataset v3 | HF Hub compatible, lingua franca |
| Training phases | I (full pretrain) → II (1% adaptation) | Mirrors X-VLA, strongest narrative |
| Vision encoder | From-scratch ViT-S | No pretrained weights dependency |
| Text encoder | From-scratch 4-layer Transformer | No pretrained weights dependency |
| Deterministic fallback | Gaussian-MSE head | Switchable via config if flow-matching underperforms |
