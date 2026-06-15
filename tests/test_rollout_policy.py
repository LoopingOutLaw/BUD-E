"""End-to-end smoke test: rollout_policy.py produces an MP4 from a real checkpoint.

This is the test gate the user wanted — the "real VLA works" claim
must produce a non-empty MP4.
"""
import os
import subprocess
import sys
from pathlib import Path


def test_rollout_produces_mp4():
    out_dir = Path("/tmp/test_rollout_vla")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / "pick_vla_rollout.mp4"
    if out_mp4.exists():
        out_mp4.unlink()

    ckpt = "/home/aditya/bude_vla/checkpoints/pick_224/pick_224_final.pt"
    if not Path(ckpt).exists():
        import pytest
        pytest.skip(f"checkpoint not found: {ckpt}")

    env = os.environ.copy()
    env.update({
        "PYTHONPATH": "src",
        "MUJOCO_GL": "egl",
    })

    result = subprocess.run(
        [
            sys.executable, "scripts/rollout_policy.py",
            "--ckpt", ckpt,
            "--out", str(out_mp4),
            "--num-rollouts", "1",
            "--img-size", "224",
            "--max-tries", "2",
            "--max-steps-per-try", "100",
        ],
        cwd="/home/aditya/bude_vla",
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"rollout failed (rc={result.returncode}): {result.stderr[-2000:]}"
    )
    assert out_mp4.exists(), f"MP4 not written: {out_mp4}"
    assert out_mp4.stat().st_size > 1000, (
        f"MP4 too small ({out_mp4.stat().st_size} bytes), probably empty"
    )
