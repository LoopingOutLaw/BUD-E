# Debug Journal — arm-passes-through-ball-and-table grasp bug

Started: 2026-06-19 (Asia/Calcutta)
Goal: Diagnose and fix the bug where the SO-101 arm approaches the ball
sideways, fails to attach, runs *through* the ball, sometimes through the
bowl/floor, and ends up with the ball rolling away — even when the verify
harness reports 7/10 pass. The user's criterion is *physical* contact, not
the simulator's "success" metric, which counts drops at target without
ever attaching.

## Environment snapshot (Phase 0)
- Runtime: CPython 3.12.3
- Launcher: /home/aditya/venv-bude/bin/python (the jax-enabled venv)
- Entry: scripts/verify_grasp_fix.py --grasp --episodes N --diagnose
- Ports: none — pure CPU simulation, EGL offscreen rendering
- Git HEAD: bec7500edd0676f656ef905867becda966b7a48c
- Working tree: 2 source files modified (ik.py, scripted_pick_and_place.py):
  - ik.py: added optional orientation constraint params to _ik_core
  - scripted_pick_and_place.py: GRASP phase passes orientation constraint
    jaw +Z horizontal toward ball, with ori_weight=3.0
  - Verification: REGRESSION (7/10 → 0/10 attach)
- Ball/bowl world layout (from mj_forward snapshot):
  - ball (cube body 9): world pos [0.30, 0.0, 0.030]   z=0.030 not 0.0125
  - pick_bowl (body 8): world pos [0.30, 0.0, 0.016]   rim ~14 mm above ball bottom
  - bowl (body 11 target zone): world pos [0.30, 0.40, 0.016]
  - BALL_RADIUS = 0.0125, GROUND_Z = 0.0295
- URDF body names confirmed:
  - moving_jaw_so101_v1 (id 7) — the actual moving-jaw body name
  - gripper (id 6), cube (id 9), jaw_contact site (id 3), gripperframe site (id 2)
- Jaw local axes (default pose, from mj_forward):
  - local +X → world ≈ [-0, -0.05, 0.998]   ≈ +Z (jaw-open direction)
  - local +Y → world ≈ [-1.0, 0, -0]        world −X (length axis, runs along jaw)
  - local +Z → world ≈ [-0, -0.998, -0.05]  world −Y (perpendicular to jaw, "thickness")
- References read this session:
  - references/methodology/00-setup.md, 02-investigate.md, 06-fix.md
  - references/runtimes/python.md
- Untracked, NOT TO BE COMMITTED by me (pre-existing): urdf/so101_*,
  results/so101_*.png, src/bude_vla/data/__pycache__/

## Hypotheses
1. [OPEN] The IK target z=GROUND_Z (ball center, z=0.0295) is too low
   relative to the gripper geometry. With jaw tip in the same XY as ball
   center but z=0.0295 to a 14mm-tall gripper seat, the jaw body actually
   extends down through the ball and into the bowl at IK convergence. The
   arm MUST ARREST at the gripper frame, not at the jaw tip — that's why
   the gripper SEATS INTO the table. Distinguishing evidence: at IK
   convergence, what is gripper.zpos vs. table z? If still on table, the
   bug is purely the target; if gripper is also penetrating, the arm pose
   is wrong. Fix: target above ball top surface, e.g. z = 0.045–0.050.
2. [OPEN] The gripper's "jaw_contact" site is CANNOT physically close on
   the ball because the gripper's two jaws are aligned with the wrong
   axis (we saw local +Y = −X world, i.e. running horizontally). For an
   EQUATORIAL squeeze, the jaw-open axis must point DOWN (gravity axis),
   but in the default pose +X ≈ +Z up = jaw points UP instead of sideways
   at the ball. The arm is closing the yaw motor against the ball's underside.
   Distinguishing evidence: locate the second/finger jaw body (the static
   one), and confirm whether jaw-open axis (jaws spread perpendicular)
   is vertical or horizontal at IK convergence.
3. [OPEN] The baseline "7/10 attach" was an artifact of the contact gate
   being satisfied on the FLOOR/GROUND geometry, not the ball — the
   `jaw_contact` site is on the moving finger, but MuJoCo contact detection
   between moving_jaw and any body (floor? ball?) was being credited as
   attach. Watch the diag log: jaw_qpos rises from 0.475 (mech spec) to
   ~1.8 (sprung open by ground contact) — that's the actuator being
   PUSHED OPEN by the floor while still advancing, triggering the "closed"
   predicate by oscillating. Distinguishing evidence: at the moment grasp
   announces attached=True, dump data.contact[*].geom1/geom2 — confirm
   whether it's ball–moving_jaw or table–moving_jaw.
4. [OPEN] The orientation-constraint patch makes things worse because the
   chosen `target_axis=[0,0,1]` (jaw +Z) on a gripper whose +Z is the
   THICKNESS axis (perpendicular to the open jaw) can't be aligned to
   "horizontal toward ball" by any 5-DOF arm without dislocating the
   wrist. The unconstrained solver was already finding this conformation
   because the seed closed-jaw qpos made +Y≈–X world naturally. Causal
   evidence (already captured): ori_weight sweep 0.5/1/2/3 → gap
   8.6/12.8/35.6/110.4 mm when constrained on +Y; gap 261 mm when on +Z.

## Failed hypothesis round counter
- Round 0 (this turn): no fix attempted; hypothesis formation only.

## Artifacts to revert
- [ ] /home/aditya/bude_vla/.git/info/exclude: added `.debug-journal.md` — KEEP (already per-clone, non-committed)
- [ ] env: unset PYTHONBREAKPOINT — KEEP unset, no breakpoints set
- [ ] /tmp/diag_*.log: pre-existing debug prints, will leave alone unless I add to them
- [ ] WIP edits to src/bude_vla/ik.py and src/bude_vla/scripted_pick_and_place.py — REVERT before commit
- [ ] ~/BUD-E/master BUD-E env: untouched this session (only bude_vla has the WIP)
- [ ] /home/aditya/bude_vla/scripts/debug_grasp_hypotheses.py — probe script. Revert: `git checkout scripts/debug_grasp_hypotheses.py && rm scripts/debug_grasp_hypotheses.py`
- [ ] /home/aditya/bude_vla/scripts/record_debug_video.py — debug recorder. Revert: `git checkout scripts/record_debug_video.py && rm scripts/record_debug_video.py`

## Findings
### 2026-06-19 — jaw closing pulse during dikwave on a gap of 240 mm
- Source: scripts/diag_attach_gates.py observed trajectory (seed 7)
- Value: jaw_qpos oscillates [1.474, 1.516, 1.691, 1.800, 1.829] during
  step 50–95 while gap stays [154, 173, 225, 251, 252, 244, 244] mm
- Interpretation: jaw actuator is being pushed OPEN by external contact
  (every push of ctrl toward JAW_CLOSED bumped by a contact spring back).
  The contact is between moving_jaw and EITHER the bowl rim, the bowl
  inside, or the table — NOT the ball (gap is 240 mm away from the ball).
- Refutes/Confirms: H3 strongly indicated (attach is satisfied by non-ball contact),
  also H1 indicated (arm overshoots into the bowl/table).
