# Real VLA Inference on Pick-and-Place — Design Spec

Date: 2026-06-15
Status: brainstorming → spec

## Goal

Demonstrate a **real** BUD-E trained VLA policy that *observes a cube and a
target zone in image input* and *runs end-to-end control of the simulated
UR5e arm to pick up the cube and place it at the target*. The user can
"place a cube anywhere on the table, press play, watch the policy do it."

If a single try fails, the policy **resets its arm to home pose and tries
again** (up to N tries) — like a real picking policy under error recovery.

## Current state (Stage 1)

We have:
- `ScriptedPickAndPlace` — 100% reliable scripted reference (file:
  `src/bude_vla/scripted_pick_and_place.py`)
- 100 episodes of *scripted demo data* recorded at 64×64, free camera
  (`data/pick_v3/`)
- A trained `BUDEPolicy` (5k steps, loss 0.35 → 0.11,
  `checkpoints/pick/pick_final.pt`)
- `render_pick_rollout.py` that runs the **scripted** policy and outputs
  `demos/videos/pick_rollout.mp4`

But:
- **Network inference is never closed-loop.** No script loads the
  checkpoint + drives the sim with `policy.sample()`.
- **Image input is 64×64.** Too small for the network to visually
  localize a cube on a busy checker-pattern table.
- **Action labels in training data are motor velocity commands**, but
  the actual arm motion in the recorded demos is **kinematic
  `qpos[7:13]` override** — so the network learns an action that doesn't
  correspond to any physical command during inference. Inconsistent.
- **No retry / reset on failure.**

## Approach (the agreed plan)

1. **Re-record pick demos at 224×224 from the free camera**, with **the
   kinematic arm target as the action label** (instead of the motor ctrl).
   - Action vector: `data.qpos[7:13]` concatenated with `data.ctrl[6]`
     (gripper) → 7-dim, semantically: "go to this arm pose with this
     gripper command."
   - Proprio: keep `data.qpos[7:15]` (current arm pose + gripper), 8-dim.
     **No cube position in proprio** — the network must localize the
     cube visually.
2. **Retrain policy** at 224×224 input (the existing ViTSmall supports
   it; `img_size=224` cfg already exists).
3. **Write `rollout_policy.py`** that:
   - Loads checkpoint.
   - Spawns the sim, places the cube at a random (x, y) on the table.
   - Loops: render image → run `policy.sample()` → get 7-dim action chunk →
     use first action's qpos target to kinematic-override arm → step sim.
   - **On failure** (cube falls off, EE drifts, N steps exceeded
     without reaching target zone): reset arm + gripper to home, jump
     the policy to phase 0, try again.
4. **Output**: `demos/videos/pick_vla_rollout.mp4` showing 1+ tries per
   cube placement, with overlay text "try 1/3", "try 2/3", etc. If a try
   succeeds, mark it "SUCCESS"; if all tries fail, mark it "FAILED".

## Components

### Component A: Re-recording script (modify `record_pick_episodes.py`)

Modify `_main_loop` to:

- Take `--img-size 224` (default).
- Take `--action-mode kinematic` (default; record arm target qpos).
- Use free camera (already supported after today's fix).

The `actions` written to parquet become `np.concatenate([arm_target,
[gripper_ctrl]])`, where `arm_target = policy.step(...)[1]` and
`gripper_ctrl = policy.step(...)[0][6]`.

### Component B: Dataset reader update

`BUDETrainingDataset` already reads `observations: (T, 8)` and
`actions: (T, 7)`. No changes needed IF we keep owning the same shape.
But we record at 224×224 now, so the precached `all_images.npy` will
have shape `(T, 224, 224, 3)` not `(T, 64, 64, 3)`. The `read()` method
hardcodes the small image cache shape and the dataset normalizes with
`/255` on whatever is there. So no code change, just regeneration.

### Component C: Training run

```bash
unset PYTHONPATH
PYTHONPATH=src python scripts/train.py \
    --data-root /home/aditya/bude_vla/data/pick_v3_224 \
    --task pick \
    --n-steps 10000 \
    --save-every 2000
```

(Starting fresh dataset dir `pick_v3_224` because shape changes.)

### Component D: `scripts/rollout_policy.py` (new file)

The core of the spec. Pseudocode:

```
load model
spawn simulation, place cube at (cx, cy) on table
reset arm to home pose
for try in range(MAX_TRIES):
    reset arm to home
    set policy internal state phase = 0
    for step in range(STEP_LIMIT):
        render image (224x224)
        build batch {images, text_ids, proprio, domain_id}
        action_chunk = policy.sample(batch)  # (1, chunk, 7)
        a = action_chunk[0, 0, :].cpu().numpy()  # first action
        arm_target = a[0:6]
        gripper_ctrl = a[6]
        data.ctrl[:] = 0
        data.ctrl[6] = gripper_ctrl
        data.qvel[6:12] = 0
        data.qpos[7:13] = arm_target
        mujoco.mj_step(model, data)
        # check termination: cube distance to target
        if cube.within 0.10 of target:
            mark success
            render success frame
            break
    if not success this try:
        reset arm to home pose
# write MP4 with frames from each try, overlay "try N/M", success/fail
```

Important details:

- **Cube carry during rollout**: replicate `_carry_cube_with` logic from
  the scripted policy. Detect when the gripper has reached the cube and
  "attach" it. If the network's arm target diverges from the cube's
  position, release the carry and let the cube fall. On retry, reset
  arm home + cube to its placed position.

- **Cube state reset**: between tries, set `data.qpos[0:7]` back to
  the cube's initial placement.

- **Action chunk stepping**: use only the first predicted action per
  step; ignore chunk shortcutting (chunk=4 is fine, we just take [0]).

### Component E: Failure detection

We need to know if the policy is *trying* but failing (vs. did nothing).
Triggers for "this try failed":

1. STEP_LIMIT steps elapse without the cube within 0.10 of target.
2. The arm has gone far outside workspace (any joint out of ±π).
3. The cube committed to extreme z (below table top - 0.05 OR above
   1.5m).
4. The cube position becomes NaN.

If a try succeeds → overall SUCCESS. If all tries fail → overall FAILED.

## Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Network drifts off rails | Detect by joint limits / EE distance from cube |
| Cube slips from "gripper" because kinematic carry isn't engaged | Use the same `_carry_cube_with` logic the scripted policy uses |
| Network predicts arm target with NaNs | clamp predicted actions; replace with home pose |
| Network never closes gripper | gripper is independent of arm, so if network decides `a[6]` is open, the cube never gets attached |
| Cube falls when network tries to move before gripper engages | gripper engage condition: EE distance to cube < 0.04 AND `a[6] < 0` (close) for 3 consecutive frames |

## Testing / verification

1. **Unit-ish test for rollout**: write
   `tests/test_rollout_policy.py` that loads the same checkpoint, runs
   the rollout loop on 3 different cube positions, asserts that at
   least 1 of 3 tries succeeds, asserts MP4 file is written. This makes
   the "real VLA works" claim testable and gates any "demo is ready"
   statement.

2. **Visual inspection**: render the MP4, confirm:
   - arm tries at least once
   - retry is visible (overlay text "try 1", "try 2")
   - either SUCCESS or honest FAILED visible at end

3. **Reliability sample**: rerun rollout on 5 random start positions;
   verify at least 1/5 reaches target *via the network*, not just
   scripted fallback.

## Success criterion

The user can:
- Run `PYTHONPATH=src python scripts/rollout_policy.py --ckpt
  /home/aditya/bude_vla/checkpoints/pick/pick_v3_224_final.pt --num-rollouts 5`
- See a single MP4 (`demos/videos/pick_vla_rollout.mp4`) where 5 trials
  show arms trying to pick up the cube. Even if only 60% trials
  succeed, that is honest "trained network doing pick-and-place most of
  the time" and the user can post it.

## Out of scope (Stage 3+)

- Multi-cube / multi-target — explicit next step.
- Real-world camera + IK in the loop instead of MuJoCo sim.
- Vision-only proprio removal — we keep the 8-dim proprio so the
  network has pose info besides the camera.
- Online learning / fine-tuning per trial.
- Curriculum over reach → push → pick.

## Risks

- **Vision-only cube pickup at 224×224 from free cam may still fail**.
  If after 10k steps the network doesn't generalize, we'd need to add
  cube_xyz to the proprio (you asked not to, but re-discuss if it
  fails).
- **Action-head consistency**: if the action head overfits to training
  data distribution, it'll output near-zero actions on init phase →
  arm doesn't move. Will detect this in early eval and retrain.
- **Reset logic may be tricky** if the policy itself gets confused by
  reset (its proprio will jump from end state to home state). Will
  need to ensure the home pose is well-conditioned in training data
  (which it is — every episode starts from home).
