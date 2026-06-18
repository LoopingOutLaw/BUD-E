"""Smoke-train test: 3 forward+backward steps on a tiny batch to prove the
   trainable graph is wired end-to-end on GPU."""
import torch
from bude_vla.models.policy import BUDEPolicy, BUDEConfig


def test_train_step_runs_on_gpu():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA required for this smoke test")
    cfg = BUDEConfig()
    p = BUDEPolicy(cfg).to("cuda")
    opt = torch.optim.AdamW(p.parameters(), lr=1e-4)
    B = 2
    imgs = torch.zeros(B, 6, cfg.img_size, cfg.img_size, device="cuda")
    qpos = torch.randn(B, cfg.state_dim, device="cuda")
    text_ids = torch.randint(0, 100, (B, 8), device="cuda")
    domain_ids = torch.zeros(B, dtype=torch.long, device="cuda")
    actions = torch.randn(B, cfg.chunk_size, cfg.action_dim, device="cuda")
    tau = torch.rand(B, device="cuda")
    noise = torch.randn_like(actions)
    batch = dict(images=imgs, text_ids=text_ids, proprio=qpos,
                  domain_id=domain_ids, actions=actions, tau=tau, noise=noise)

    losses = []
    for _ in range(3):
        out = p(batch)
        target = actions - noise
        loss = ((out["velocity"] - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())

    # Loss should decrease at least slightly across 3 steps (flow-matching is easy)
    assert losses[-1] < losses[0] - 1e-4, f"loss did not decrease: {losses}"


def test_sample_runs_on_gpu():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA required for this smoke test")
    cfg = BUDEConfig()
    p = BUDEPolicy(cfg).to("cuda").eval()
    B = 2
    with torch.no_grad():
        a = p.sample(dict(
            images=torch.zeros(B, 6, cfg.img_size, cfg.img_size, device="cuda"),
            text_ids=torch.randint(0, 100, (B, 8), device="cuda"),
            proprio=torch.randn(B, cfg.state_dim, device="cuda"),
            domain_id=torch.zeros(B, dtype=torch.long, device="cuda"),
        ))
    assert a.shape == (B, cfg.chunk_size, cfg.action_dim)
