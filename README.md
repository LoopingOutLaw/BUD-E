# BUD-E

A 22M-parameter, soft-prompted, flow-matching Vision-Language-Action model,
re-implemented from scratch with reference to X-VLA (Zheng et al., 2025).

**Status:** In development. See `docs/superpowers/specs/2026-06-13-bude-vla-design.md`
for the design and `docs/architecture.md` for the architecture writeup.

## Quick start

```bash
python3 -m venv .bude-venv
. .bude-venv/bin/activate
pip install -e ".[dev,sim]"
pytest
```

## Architecture (TL;DR)

```
Camera ─▶ ViT-S  ──────────┐
Text    ─▶ BPE+Transformer ┤──▶ Transformer Backbone ──▶ Flow-Matching ─▶ actions
State   ─▶ Linear ─────────┤    ▲
                            │    │
Soft prompts (1 per domain) ─┘    │
```

Five from-scratch components, ~22M params, soft prompts encode the robot/task,
Phase II adapts only ~1% of params to a new task.

## Inspiration

Built with reference to:
- **X-VLA** (Zheng et al., 2025) — soft prompts + flow matching on cross-embodiment data
- **SmolVLA** — small VLA recipes
- **Octo** — unified transformer policy

## Citation

Built by Aditya, 2026.
