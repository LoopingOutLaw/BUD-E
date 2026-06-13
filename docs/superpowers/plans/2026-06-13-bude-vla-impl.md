

---
## Task 1: Bootstrap the package

**Files:**
- Create: `pyproject.toml`
- Create: `src/bude_vla/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_bootstrap.py`
- Create: `.gitignore`
- Create: `README.md`
- Create: `LICENSE`

- [ ] **Step 1.1 — Write the failing test**

Create `tests/__init__.py` (empty) and `tests/test_bootstrap.py`:

```python
def test_package_imports():
    import bude_vla
    assert bude_vla.__version__ == "0.1.0"
```

- [ ] **Step 1.2 — Run the test to verify it fails**

```bash
cd /home/aditya/bude_vla
pip install -e ".[dev]" 2>&1 | tail -5
pytest tests/test_bootstrap.py -v
```

Expected: `FAILED tests/test_bootstrap.py::test_package_imports — ModuleNotFoundError: No module named 'bude_vla'`

- [ ] **Step 1.3 — Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "bude_vla"
version = "0.1.0"
description = "A 22M-parameter soft-prompted flow-matching VLA, re-implemented from scratch"
readme = "README.md"
license = {file = "LICENSE"}
requires-python = ">=3.11"

dependencies = [
    "torch>=2.4.0",
    "numpy>=1.26",
    "pillow>=10.0",
    "pyyaml>=6.0",
    "huggingface-hub>=0.24",
    "tokenizers>=0.19",
    "pyarrow>=16",
    "imageio>=2.34",
    "imageio-ffmpeg>=0.5",
    "wandb>=0.18",
]

[project.optional-dependencies]
sim = [
    "mujoco>=3.2.0",
    "mujoco-mjx>=3.2.0",
    "jax[cuda12]>=0.4.30",
]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.6",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q"
```

- [ ] **Step 1.4 — Write the package init**

Create `src/bude_vla/__init__.py`:

```python
"""BUD-E: A 22M-parameter soft-prompted flow-matching VLA."""

__version__ = "0.1.0"
```

- [ ] **Step 1.5 — Write `.gitignore`**

```
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.coverage
build/
dist/
*.egg-info/
.venv/
venv/
wandb/
checkpoints/
datasets/
*.mp4
*.parquet
.DS_Store
```

- [ ] **Step 1.6 — Write the LICENSE (Apache-2.0)**

Create `LICENSE` containing the Apache License 2.0 text. Use any standard Apache-2.0 LICENSE body (replace `Copyright [year] [name]` with `Copyright 2026 Aditya`). Full text is at https://www.apache.org/licenses/LICENSE-2.0.txt — paste verbatim from that URL.

- [ ] **Step 1.7 — Write minimal README.md**

```markdown
# BUD-E

A 22M-parameter soft-prompted flow-matching Vision-Language-Action model, re-implemented from scratch with reference to X-VLA.

**Status:** In development. See `docs/superpowers/specs/2026-06-13-bude-vla-design.md` for the design.

## Quick start

```bash
pip install -e ".[dev,sim]"
pytest
```

## Citation

Built by Aditya, 2026. Architecture inspired by X-VLA (Zheng et al., 2025) — see paper at arxiv.org/abs/2510.10274.
```

- [ ] **Step 1.8 — Install and run test**

```bash
cd /home/aditya/bude_vla
pip install -e ".[dev]" 2>&1 | tail -10
pytest tests/test_bootstrap.py -v
```

Expected: `1 passed`. If PyTorch install fails, comment out the `torch` dep temporarily and reinstall — we'll add it in Task 2.

- [ ] **Step 1.9 — Commit**

```bash
cd /home/aditya/bude_vla
git init
git add .
git commit -m "Task 1: bootstrap bude_vla package"
```

---

## Task 2: Vision tower (ViTSmall)

**Files:**
- Create: `src/bude_vla/models/__init__.py`
- Create: `src/bude_vla/models/vision.py`
- Create: `tests/test_vision.py`

- [ ] **Step 2.1 — Write the failing test**

Create `src/bude_vla/models/__init__.py` (empty) and `tests/test_vision.py`:

```python
import torch
from bude_vla.models.vision import ViTSmall


def test_vitsmall_output_shape():
    model = ViTSmall(img_size=224, patch_size=16, in_channels=3, dim=256,
                     depth=2, heads=4, mlp_ratio=2.0)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 196, 256)  # 14*14 patches, projected dim


def test_vitsmall_param_count_below_budget():
    model = ViTSmall(img_size=224, patch_size=16, in_channels=3, dim=384,
                     depth=12, heads=6, mlp_ratio=4.0)
    n = sum(p.numel() for p in model.parameters())
    # budget is 5M per spec; allow 20% overshoot
    assert n < 6_000_000, f"ViT-S too large: {n} params"
    assert n > 1_000_000, f"ViT-S too small: {n} params"


def test_vitsmall_accepts_different_image_size():
    model = ViTSmall(img_size=128, patch_size=16, in_channels=3, dim=64,
                     depth=1, heads=2, mlp_ratio=2.0)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 64, 64)  # 8*8 patches
```

- [ ] **Step 2.2 — Run test to verify it fails**

```bash
cd /home/aditya/bude_vla
pytest tests/test_vision.py -v
```

Expected: `ModuleNotFoundError: No module named 'bude_vla.models.vision'`

- [ ] **Step 2.3 — Implement `ViTSmall`**

Write `src/bude_vla/models/vision.py`:

```python
"""From-scratch ViT-small vision tower for BUD-E."""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)              # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class ViTSmall(nn.Module):
    """From-scratch Vision Transformer (no pretrained weights).

    Output: (B, num_patches, out_dim) where out_dim is the model's projection dim.
    Spec target: out_dim=256 (after a final linear from inner dim=384).
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        dim: int = 384,
        depth: int = 12,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        out_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, dim)
        n_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        # Project to the model's common dim (256 per spec).
        self.proj = nn.Linear(dim, out_dim) if out_dim != dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.proj(x)
        return x
```

- [ ] **Step 2.4 — Run test to verify it passes**

```bash
cd /home/aditya/bude_vla
pytest tests/test_vision.py -v
```

Expected: `3 passed`.

- [ ] **Step 2.5 — Commit**

```bash
cd /home/aditya/bude_vla
git add src/bude_vla/models/vision.py src/bude_vla/models/__init__.py tests/test_vision.py
git commit -m "Task 2: ViT-S vision tower"
```
