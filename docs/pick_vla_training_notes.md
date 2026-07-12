# SO-101 Pick VLA: V37 Reset and V38 Continuation

Date: 2026-07-12

This is the authoritative experiment note for the current pipeline. Commands
for v26-v36 are intentionally not retained here because their camera, feature,
and action-rate contracts are incompatible with v37.

## Non-Negotiable Observation Contract

The autonomous policy may use:

- top RGB and wrist RGB;
- robot joint and gripper encoder values;
- a natural-language instruction;
- features computed from camera pixels.

It may not use simulator cube coordinates, a cube-to-gripper vector, episode
progress, or a simulator-only phase label. Simulator object state is allowed
for generating expert labels and measuring outcomes, but it is not part of the
learned policy batch. V37 records `state_dim=6` to enforce this boundary.

## Why More Old Training Was Not The Answer

The pre-reset random benchmarks were:

| Checkpoint | Any contact | Strict grasp | Success |
| --- | ---: | ---: | ---: |
| v26 | 6/100 | 1/100 | 0/100 |
| v30 | 9/100 | 0/100 | 0/100 |
| v31 | 27/100 | 0/100 | 0/100 |
| v32 | 17/150 | 0/150 | 0/150 |
| v33 | 45/150 | 0/150 | 0/150 |
| v34 | 19/60 | 0/60 | 0/60 |
| v35 | 1/150 | 0/150 | 0/150 |

The v33 intervention dataset itself contained 1,480 episodes and more than 50%
strict-grasp frames, yet the autonomous policy still produced 0/150 strict
grasps. That contradiction showed that demonstration count and terminal-frame
weighting were not the primary fault. The observation/action contract had to be
audited before collecting more data.

## Confirmed Root Causes

### 1. The top camera did not cover the claimed workspace

The old `front_top` camera was pitched toward positive Y. At `y=-0.10`, cube
placements across different X values produced the same apparent red centroid.
The learned policy could not infer a unique target from those images.

Fix:

- true overhead camera at `[0.25, 0.03, 0.80]`;
- camera axes aligned with the table;
- 45 degree vertical field of view;
- validated operational workspace `x=[0.22, 0.34]`, `y=[-0.03, 0.06]`.

### 2. Red debug geometry was a false cube detector

The static fingertip site and pad were rendered bright red/pink. The old
all-red-pixel average could move toward the arm as the gripper entered view.

Fix:

- fingertip sites made transparent and neutral;
- contact pads made gray;
- red segmentation changed to connected components;
- runtime tracking associates one component through time and rejects large
  jumps or out-of-workspace detections.

### 3. Demonstration and rollout action rates disagreed

The teacher changed control targets every four 2 ms physics steps, which is
125 Hz. Videos were marked near 30 FPS, and rollout treated each stored action
as a new 125 Hz action. This created long runs of almost identical labels and
made the learned freeze behavior likely.

The same successful trajectories were replayed after decimation:

| Record stride | Effective rate | Replay success |
| ---: | ---: | ---: |
| 2 | 62.5 Hz | 29/30 |
| 4 | 31.25 Hz | 29/30 |
| 6 | 20.8 Hz | 27/30 |

V37 uses stride 4. The expert still solves IK at 125 Hz and every fourth target
forms a candidate action plan. That plan is then executed from reset at the
exact 31.25 Hz deployment rate; images and proprio are recorded from this
second pass, not from the decimated high-rate states. Any plan that fails the
policy-rate replay is discarded. A learned action is likewise held for 16
physics substeps, so stored transitions and rollout now have one contract.

This is consistent with the official OpenVLA warning that unnecessarily
high-frequency, nearly idle action data can make policies stall, while action
chunking permits a higher practical command rate than single-action OpenVLA.

### 4. DINOv2 consumed the wrong history frame

With four-frame dual-camera history, the pretrained RGB patch kernel was copied
to channels 0:3, the oldest top frame. The current top image was near the end of
the channel stack. The input also skipped the normalization expected by the
pretrained backbone.

Fix:

- initialize the pretrained patch kernel on the current top RGB group;
- initialize history/wrist channel weights to zero for learned adaptation;
- apply ImageNet mean/std normalization independently to every RGB group;
- reduce active history to two frames to limit RAM and temporal ambiguity.

### 5. The expert encoded invalid recovery transitions

Three state-machine defects polluted recovery data:

- lift/move used the original cube spawn after a retry had displaced it;
- a one-frame two-pad contact could permanently count as a successful grasp;
- lift interpolation reached only 75% before switching phase and jumping.

Fix:

- capture the live grasp anchor before lift;
- require recent sustained contact through tightening;
- terminate failed close attempts instead of performing an empty place;
- complete lift interpolation over the configured lift duration;
- allow camera tracker reacquisition only after backing away for a retry.

### 6. Evaluation retained frames it never used

Closed-loop training eval appended every 224px history stack to the result even
when no video was requested. Combined with 14 GiB caches, this caused RAM spikes
and contributed to desktop instability.

Fix:

- non-video eval stores no frames;
- v37 cache is 6,000 frames with two-history input, about 3.4 GiB;
- batch size 4, accumulation 8, one worker;
- free-disk and available-RAM checks before expensive stages;
- local `TMPDIR` so a full system `/tmp` does not break PyTorch imports.

## Independent System Validation

The new visual-servo benchmark calibrates a fixed top camera using known table
points, then controls the expert from RGB-derived cube XY. Runtime control does
not read MuJoCo cube pose; true pose is read only for error/success reporting.

Safe-workspace result, seed 777:

```text
episodes: 50
success episodes: 50/50 (1.000)
any_contact episodes: 50/50 (1.000)
strict_grasp episodes: 50/50 (1.000)
median initial localization error: 0.32 mm
p95 initial localization error: 0.52 mm
median pre-grasp tracking error: 0.32 mm
p95 pre-grasp tracking error: 2.58 mm
```

This establishes a 100% observed task ceiling for the chosen 50-position sample
without privileged runtime cube state. It does not establish learned-policy
success; that is measured separately after training.

A fresh 19-episode 224px/state6/stride4 dataset was then read back from disk and
its persisted controls replayed at 16 substeps per action. Result: 19/19 strict
grasps and 19/19 successful places.

## V37 Dataset Recipe

The full recorder runs 4,000 randomized attempts in the validated workspace.
Only successful episodes are written.

Approximate mixture:

| Behavior | Probability/configuration |
| --- | --- |
| clean strategy | about 70% after independent perturbation probabilities |
| recoverable XY/Z waypoint error | 20%, max 3 mm XY and 2 mm Z |
| light nudge then recovery | 5%, max 3 mm XY and 2 mm Z |
| first-close miss then retry | 8%, max 4 mm XY, one retry |

These values are intentionally smaller than the old 8-18 mm perturbations.
The previous curriculum often changed the task distribution more than it taught
local recovery. V37 preserves a dominant, consistent expert mode and adds only
recoverable local variation.

Required gates:

1. Camera-only expert success at least 95% over 100 episodes.
2. At least 3,200 successful written demonstrations.
3. Persisted-action replay success at least 95% over 200 random episodes.
4. Cache files exist and are non-empty before training.

## V37 Training Recipe

V37 starts from DINOv2 visual weights but from no prior BUD-E checkpoint.

```text
image/history:       224px, top+wrist, 2 times
state:               6D joints/gripper
action:              6D absolute targets, chunk 16
sampled cache:       6000 frames, phase-balanced
batch:               4 x accumulation 8 = effective 32
training horizon:    25000 microsteps (~3125 optimizer updates)
new-module LR:       1e-4 cosine schedule
backbone LR:         1e-6
BC / flow weight:    8.0 / 0.10
gripper dimension:   5x
early / late BC:     4x through 0.25 / from 0.35
EMA:                 0.999
```

The original runner evaluated 12 episodes every 5,000 steps with temporal
ensembling and EMA weights. That evaluation mode was later proven incorrect for
this checkpoint; all scores tied at zero, so the old strict-greater comparison
left the 5,000-step file as `best.pt`.

## V37 Evaluation Audit

The completed dataset contains 3,784 replay-verified successful episodes. The
first cache contained 6,000 rows but represented only 2,986 episodes (78.9%);
798 randomized initial cube placements were absent.

The original pipeline benchmark selected the 5,000-step EMA checkpoint and
used temporal ensembling. Its 0/150 result was therefore not a valid measure of
the final model. Controlled native-chunk tests gave:

| Policy / execution | Success | Contact | Strict grasp |
| --- | ---: | ---: | ---: |
| raw 25k, native chunk, 50 positions | 5/50 | 23/50 | 7/50 |
| raw 20k, native chunk, first 30 | 0/30 | 13/30 | 2/30 |
| raw 25k, horizon 8, first 20 | 0/20 | 2/20 | 1/20 |
| raw 25k plus simulator contact reflex, first 30 | 4/30 | 15/30 | 8/30 |

Consequences:

- v37 learned a real camera-conditioned policy; it is not a 0% checkpoint;
- optimization was still improving between 20k and 25k;
- EMA decay 0.999 lagged the useful raw weights;
- replanning at horizon 1 or 8 broke the coherent 16-action trajectory;
- contact-triggered closure alone did not improve end-to-end placement.

## V38 Continuation Recipe

V38 changes coverage and evaluation, not the deployable observation contract:

```text
source weights:       v37 final raw model_state_dict
sampled cache:        64000 frames
coverage floor:       12 phase-stratified rows from every episode
training horizon:     100000 microsteps
new-module LR:        2e-5 cosine schedule
backbone LR:          2e-7
EMA:                  disabled
train seed:           3802
eval:                 30 random episodes every 10000 steps
execution:            native full 16-action chunks
final benchmark:      150 random positions
```

The cache dry run produced exactly 64,000 unique rows with a minimum of 12,
median of 17, and maximum of 28 rows per episode. The runner requires 70 GiB
free before building the approximately 36 GiB memory-mapped cache.

## V38 Completed Result and V39 Diagnosis

V38 training-time evaluation peaked at 5/30 on both 70k and 90k, then declined
to 3/30 at 100k. The selected raw 90k checkpoint produced 21/150 success,
64/150 contact, and 36/150 strict grasp on seed 3805. The separate fixed eight
positions produced 0/8, so that video was not representative of global success.

Spatial bins from the 150-position benchmark:

| Cube Y | Episodes | Contact | Grasp | Success |
| --- | ---: | ---: | ---: | ---: |
| -0.03 to 0.00 | 53 | 13% | 2% | 0% |
| 0.00 to 0.03 | 40 | 45% | 25% | 20% |
| 0.03 to 0.06 | 57 | 68% | 44% | 23% |

The failure is a compressed visual-to-action slope, not missing data. For a
cube at X=0.28 moved across the full Y range:

```text
expert shoulder-pan span: 0.305 rad
v37 final span:           0.024 rad
v38 70k span:             0.080 rad
v38 90k span:             0.072 rad
v38 100k span:            0.073 rad
```

The 64k cache contains 18,323 early-phase rows with shoulder-pan slope -4.13
rad per meter of cube Y and correlation -0.989. The model has enough balanced
labels but its shared loss underweights this deployment-critical dimension.
The sensitivity improvement peaked and regressed before 100k, ruling out a
five-million-step continuation on the unchanged objective.

V39 initializes from v38 best raw weights and changes only optimization:

```text
training horizon:          60000 microsteps
new-module LR:             1e-5
backbone LR:               1e-7
shoulder-pan loss weight:  10x
gripper loss weight:       5x
BC loss weight:            8.0
flow loss weight:          0.02
EMA:                       disabled
eval:                      40 positions every 5000 steps
acceptance gate:           shoulder-pan span >= 0.14 rad
```

The 0.14-rad gate requires nearly twice v38 sensitivity while remaining below
the 0.305-rad expert reference. The selected 35k checkpoint passes at 0.142 rad
and achieved 25/100 on a fresh random benchmark. Step 45k reached 0.167 rad
but scored lower at 22/100, so 35k remains selected. A checkpoint that fails
this gate is not allowed to spend time on the 150-position final benchmark.

## V39 Completed Result

The selected step-35k checkpoint passed the recalibrated 0.14-rad Y-sensitivity
gate at 0.142 rad. Final results were:

```text
150-position success:       31/150 (20.7%)
150-position contact:       79/150 (52.7%)
150-position strict grasp:  48/150 (32.0%)
fixed-set video:             3/8 (37.5%)
```

The successful video trajectories approach from vision, close, lift, rotate
toward the target on the right, transport, and place. That rightward turn is
expected task behavior.

Spatial result:

| Region | Contact | Grasp | Success |
| --- | ---: | ---: | ---: |
| Y -0.03 to 0.00 | 17% | 9% | 6% |
| Y 0.00 to 0.03 | 62% | 44% | 32% |
| Y 0.03 to 0.06 | 75% | 42% | 23% |
| X 0.22 to 0.25 | 71% | 49% | 34% |
| X 0.31 to 0.34 | 26% | 17% | 11% |

V39 improved the Y-axis bottleneck but exposed radial under-response. Across
X=0.22 to 0.34 at fixed Y, the expert shoulder-lift target spans 0.134 rad;
v39 spans 0.007 rad with the wrong slope and partly substitutes 0.061 rad of
elbow motion. A future continuation should weight radial shoulder-lift
precision and preserve the validated shoulder-pan sensitivity. It should not
resume the unchanged objective for millions of steps.

## V40 Completed Result and Corrected Diagnosis

V40 continued from v39 with 10x shoulder-pan and 10x shoulder-lift loss
weights. Its deterministic 6x6 workspace evaluations were:

    step  5000: 4/36
    step 10000: 1/36
    step 15000: 4/36
    step 20000: 4/36
    step 25000: 5/36
    step 30000: 6/36
    step 35000: 5/36
    step 40000: 1/36
    step 45000: 8/36  selected
    step 50000: 4/36
    step 55000: 6/36
    step 60000: 2/36

The selected step-45k checkpoint reached 0.182 rad shoulder-pan span but only
0.008 rad shoulder-lift span. The old lift gate rejected it. Forward kinematics
then showed why individual-joint matching is the wrong acceptance condition:
the policy can substitute elbow motion for shoulder motion. What matters is TCP
position. Across a workspace corner grid, v40 first actions differ from expert
TCP targets by 15.8 mm median, 22.5 mm p95, and 23.5 mm maximum. Those errors
are large relative to the 30 mm cube and explain the remaining reach misses.

The conclusion is not to increase shoulder weight again. Joint-space BC asks
the small policy to learn both visual localization and nonlinear SO-101 inverse
kinematics. V40 weighting changed joint allocation without reducing task-space
error or materially improving success.

## V41 Absolute Task-Space Action Pipeline

V41 changes the output representation, not the deployable observation contract.
The VLA receives top/wrist RGB, language, and six measured joint/gripper
positions. It predicts four values per action:

    [target TCP x, target TCP y, target TCP z, gripper target]

The shared damped-least-squares IK controller converts each predicted TCP target
to legal SO-101 joint targets. Simulator cube coordinates are never policy
inputs. Source joint actions are converted offline to TCP labels with forward
kinematics; at runtime the TCP targets must still be inferred from camera
observations.

The representation was validated before any GPU training:

    camera-only expert benchmark:      100/100
    source joint-action replay:         200/200
    in-memory ee_abs oracle replay:      98/100
    production ee_abs replay gate:      199/200 (99.5%)
    converted episodes:                   3784
    converted dataset disk use:          96 MiB

The converted root symlinks the existing videos, and training reuses the
existing 64k, 36 GiB memory-mapped image cache. This avoids another large data
copy.

V41 recipe:

    initialization:             v40 best raw shared trunk
    action outputs:             fresh 4D-compatible output tensors
    training horizon:           120000 microsteps
    new-module LR:              3e-5
    DINO backbone LR:           1e-7
    BC loss weight:             8.0
    flow loss weight:           0.0
    gripper loss weight:        5.0
    early / late sample weight: 6.0 / 4.0
    history / chunk:            2 / 16
    effective batch:            32
    EMA:                        disabled
    selection:                  6x6 deterministic workspace every 5000 steps
    execution mode:               paired native-vs-first-action 36-position test
    final acceptance:           at least 80% on 200 random positions

A first-action task-space diagnostic now reports TCP median, p95, and maximum
error. It replaces the invalid rule that one chosen joint must match the
experts IK decomposition. The final random closed-loop benchmark remains the
actual acceptance test.

The action-space change is consistent with open generalist-policy work that
supports end-effector control and adaptation to new action spaces. The
hierarchical split also follows the principle that high-level visual-language
reasoning and low-level execution can be separated while retaining visual
context throughout the policy.

## Post-Training Decision Protocol

Do not decide from training loss or one video. Use the 200-position random
benchmark and separate the failure stages:

| Result | Interpretation | Next action |
| --- | --- | --- |
| contact below 50% | visual reach still fails | run image/action sensitivity and inspect top/wrist predictions; do not add grasp data |
| contact high, grasp 0% | close/alignment transition fails | add a discrete gripper trigger trained on this corrected v37 data, then run a small ablation |
| grasp high, place low | lift/transport policy fails | rebalance cache toward lift/move without changing perception |
| success rises with checkpoints | optimization is working | continue from the best checkpoint, not necessarily final |
| offline fit good, all closed-loop checkpoints 0% | architecture/control mismatch remains | train an official LeRobot ACT baseline on the same replay-validated data |

DAgger is not the next automatic step. It becomes useful only after the fresh
base policy reaches reliably. Old DAgger rounds queried corrections from a
policy whose camera and time contracts were already wrong, so those datasets
must not be mixed into v37.

ACT is the preferred baseline escalation on this RTX 4060 because LeRobot
documents it as a lightweight, low-compute action-chunking policy for precise
manipulation. SmolVLA remains a later language-generalization path, but its
official base model is much larger and its documentation stresses repeated
coverage per task variation. Neither baseline removes the replay and camera
acceptance gates.

## Commands

Full no-time-limit pipeline:

```bash
cd /home/aditya/bude_vla
bash scripts/run_v41_ee_abs.sh
```

Regression tests:

```bash
cd /home/aditya/bude_vla
MUJOCO_GL=egl PYTHONPATH=src \
  /home/aditya/venv-bude/bin/python -m unittest discover -s tests -v
```

Standalone camera-only gate:

```bash
cd /home/aditya/bude_vla
MUJOCO_GL=egl PYTHONPATH=src \
  /home/aditya/venv-bude/bin/python scripts/benchmark_visual_servo_pick.py \
  --num-episodes 100 --min-success-rate 0.95 --seed 777
```

## Primary References

- [OpenVLA performance troubleshooting](https://github.com/openvla/openvla#vla-performance-troubleshooting)
- [LeRobot ACT documentation](https://huggingface.co/docs/lerobot/act)
- [LeRobot SmolVLA documentation](https://huggingface.co/docs/lerobot/smolvla)
- [Octo generalist robot policy](https://octo-models.github.io/)
- [RT-H action hierarchies](https://rt-hierarchy.github.io/)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
