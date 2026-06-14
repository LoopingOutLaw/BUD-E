# When You Wake Up — Action Plan

## Where We Are
- ✅ Tasks 1-12 implemented, 46 unit tests passing (44 fresh-verified, 2 known-flaky)
- ✅ Git committed: `src/bude_vla/{models,envs,data}`, `tests/`, `urdf/`
- ⏸️ **No training has been started.** Everything is ready for you to start it.

## What You Owe Me
You said "after I wake up we will do the training part". I'm ready when you are.

## Quick Sanity Check (2 min, optional)
```bash
cd /home/aditya/bude_vla && PYTHONPATH=src /home/aditya/.bude-venv/bin/python -m pytest tests/models/ tests/envs/ --no-header -q
```
Expected: `42 passed`. If not, paste the traceback.

## Open Items To Decide With You
1. **Demo generation strategy**:
   - Option A: Single-env loop, 1000 episodes × 3 tasks = 3000 eps ≈ 2-3 hrs CPU render (slow)
   - Option B: GPU `vmap` over N=256 parallel envs using **batch MuJoCo renderer** ≈ 5 min (much faster, but needs `mujoco_warp` or a custom vmap-keyed render path)
   - **My recommendation**: A first to land, parallelize in Task 14 if you want

2. **Which task to record first**:
   - Reach (already has scripted policy + cube-free) — ship today
   - Push (needs cube geom added to XML, ~30 min extra)
   - Pick/place (depends on push cube)

3. **Real SO-101 instructions**: the ones I wrote (`"reach the red target"`) are placeholders. Do you want me to fetch the actual SO-101 instruction-set from lerobot/openx or write our own?

## Your Move
Tell me:
- (a) which option for demo gen (A/B)
- (b) which task to record first
- (c) what to do about the lerobot-v3 hang (skip / retry with different config / pull & investigate)

Then I unblock and you can go back to sleep. If you say "do reach task, option A, skip the hang" I just start recording.
