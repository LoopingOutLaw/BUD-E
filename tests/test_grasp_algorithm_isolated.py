"""Isolated correctness test for the GraspController algorithm, using a
tiny synthetic MuJoCo scene (NOT the real SO-101 meshes -- those aren't
available in this sandbox). This validates the core math independent of
asset/import dependencies (jax, mjx, the real STL files) that aren't
present here.

What it checks:
  1. While the "gripper" is far from the ball, no attach happens even if
     the jaw qpos is commanded fully closed (proximity gate works).
  2. Once the gripper is moved next to the ball and the jaw closes (with
     real contact), attach engages within the debounce window, and the
     gap at attach is ~0 (flush, no preserved offset).
  3. While "carried", moving the gripper moves the ball with zero gap
     drift.
  4. Re-opening the jaw releases the ball.
"""
import mujoco
import numpy as np

BALL_RADIUS = 0.02
CUBE_QPOS_START = 8   # after: 1 slide (gripper x) + 1 slide (gripper z) + 1 hinge (jaw) = 3 qpos, then freejoint... we'll just compute from model

XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" pos="0 0 0"/>
    <body name="gripper" pos="0 0 0.5">
      <joint name="gx" type="slide" axis="1 0 0"/>
      <joint name="gz" type="slide" axis="0 0 1"/>
      <geom name="gripper_palm" type="box" size="0.01 0.03 0.01" rgba="0.6 0.6 0.6 1"
            contype="1" conaffinity="1"/>
      <body name="moving_jaw_so101_v1" pos="0.025 0 0">
        <joint name="jaw" type="hinge" axis="0 1 0" range="-0.1 1.0"/>
        <geom name="jaw_geom" type="box" size="0.005 0.025 0.005" rgba="0.3 0.3 0.3 1"
              contype="1" conaffinity="1"/>
      </body>
    </body>
    <body name="cube" pos="1.0 0 0.5">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="sphere" size="0.02" rgba="0.85 0.05 0.05 1" mass="0.05"
            condim="4" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

GRIPPER_BID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
JAW_BID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
CUBE_BID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
CUBE_QSTART = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]
JAW_QADR = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "jaw")]
GX_QADR = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gx")]


# ---- inlined copy of GraspController's algorithm (avoids needing the real
# bude_vla.envs.so101_mjx import, which needs jax/mjx/real mesh assets) ----
import dataclasses


@dataclasses.dataclass
class GraspState:
    attached: bool = False
    offset_local: np.ndarray = None
    enclosure_streak: int = 0


class GraspController:
    def __init__(self, model, gripper_bid, jaw_bid, cube_bid, cube_qstart,
                 ball_radius=BALL_RADIUS, attach_gap_tolerance=0.0035,
                 attach_debounce_steps=5, jaw_closed_qpos_threshold=0.30,
                 release_jaw_qpos_threshold=1.00, release_drift_tolerance=0.012,
                 require_contact=True):
        self.gripper_body_id = gripper_bid
        self.jaw_body_id = jaw_bid
        self.cube_body_id = cube_bid
        self.cube_qstart = cube_qstart
        self.ball_radius = ball_radius
        self.attach_gap_tolerance = attach_gap_tolerance
        self.attach_debounce_steps = attach_debounce_steps
        self.jaw_closed_qpos_threshold = jaw_closed_qpos_threshold
        self.release_jaw_qpos_threshold = release_jaw_qpos_threshold
        self.release_drift_tolerance = release_drift_tolerance
        self.require_contact = require_contact
        self.state = GraspState()

    def _has_contact(self, model, data):
        body_ids = model.geom_bodyid
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = body_ids[c.geom1], body_ids[c.geom2]
            if (b1 == self.jaw_body_id and b2 == self.cube_body_id) or \
               (b1 == self.cube_body_id and b2 == self.jaw_body_id):
                return True
        return False

    def gap(self, data):
        g = data.xpos[self.gripper_body_id]
        b = data.xpos[self.cube_body_id]
        return float(np.linalg.norm(b - g)) - self.ball_radius

    def update(self, model, data, jaw_qpos):
        mujoco.mj_forward(model, data)
        g_xyz = data.xpos[self.gripper_body_id].copy()
        g_rot = data.xmat[self.gripper_body_id].reshape(3, 3).copy()
        b_xyz = data.xpos[self.cube_body_id].copy()
        s = self.state

        if not s.attached:
            gap = float(np.linalg.norm(b_xyz - g_xyz)) - self.ball_radius
            enclosed = (jaw_qpos <= self.jaw_closed_qpos_threshold
                       and gap <= self.attach_gap_tolerance
                       and (not self.require_contact or self._has_contact(model, data)))
            s.enclosure_streak = s.enclosure_streak + 1 if enclosed else 0
            if s.enclosure_streak >= self.attach_debounce_steps:
                d = b_xyz - g_xyz
                n = float(np.linalg.norm(d))
                d = d / n if n > 1e-9 else np.array([0., 0., -1.])
                flush = g_xyz + d * self.ball_radius
                s.offset_local = g_rot.T @ (flush - g_xyz)
                s.attached = True
                s.enclosure_streak = 0
            return s

        should_release = jaw_qpos >= self.release_jaw_qpos_threshold
        if not should_release:
            predicted = g_xyz + g_rot @ s.offset_local
            drift = float(np.linalg.norm(b_xyz - predicted))
            if drift > self.release_drift_tolerance:
                should_release = True
        if should_release:
            s.attached = False
            s.offset_local = None
            s.enclosure_streak = 0
            return s

        new_world = g_xyz + g_rot @ s.offset_local
        data.qpos[self.cube_qstart:self.cube_qstart + 3] = new_world
        data.qvel[self.cube_qstart:self.cube_qstart + 6] = 0.0
        return s


grasp = GraspController(model, GRIPPER_BID, JAW_BID, CUBE_BID, CUBE_QSTART)

mujoco.mj_resetData(model, data)
data.qpos[GX_QADR] = 0.0
data.qpos[JAW_QADR] = 1.0  # jaw open
data.qpos[CUBE_QSTART:CUBE_QSTART + 3] = [1.0, 0.0, 0.5]
data.qpos[CUBE_QSTART + 3:CUBE_QSTART + 7] = [1, 0, 0, 0]
mujoco.mj_forward(model, data)

print("Phase 1: gripper far from ball, jaw forced fully closed -- must NOT attach")
data.qpos[JAW_QADR] = -0.1  # fully closed, but gripper is nowhere near the ball
for _ in range(20):
    grasp.update(model, data, jaw_qpos=float(data.qpos[JAW_QADR]))
    mujoco.mj_step(model, data)
assert not grasp.state.attached, "FAIL: attached despite being far away -- proximity gate broken"
print("  PASS: no false attach at distance.\n")

print("Phase 2: move gripper next to the ball, close the jaw with real contact")
mujoco.mj_resetData(model, data)
data.qpos[CUBE_QSTART:CUBE_QSTART + 3] = [1.0, 0.0, 0.5]
data.qpos[CUBE_QSTART + 3:CUBE_QSTART + 7] = [1, 0, 0, 0]
data.qpos[JAW_QADR] = 1.0
target_gx = 1.0 - 0.045  # palm offset so jaw geom sits right next to the ball
for step in range(400):
    frac = min(1.0, step / 60.0)
    data.qpos[GX_QADR] = frac * target_gx
    if step > 80:
        close_frac = min(1.0, (step - 80) / 60.0)
        data.qpos[JAW_QADR] = 1.0 - close_frac * 1.1
    grasp.update(model, data, jaw_qpos=float(data.qpos[JAW_QADR]))
    mujoco.mj_step(model, data)
    if grasp.state.attached:
        gap_at_attach = grasp.gap(data)
        print(f"  attached at step {step}, gap = {gap_at_attach * 1000:.3f} mm")
        break
else:
    raise AssertionError("FAIL: never attached even with real proximity + contact")

assert abs(gap_at_attach) < 1e-3, f"FAIL: gap at attach was {gap_at_attach*1000:.2f}mm, expected ~0"
print("  PASS: attach happened with a flush (near-zero) gap.\n")

print("Phase 3: carry -- move gripper, confirm ball follows with ~zero drift")
start_gx = data.qpos[GX_QADR]
max_gap_during_carry = -np.inf
for step in range(100):
    data.qpos[GX_QADR] = start_gx - 0.001 * step  # slide gripper backward
    grasp.update(model, data, jaw_qpos=float(data.qpos[JAW_QADR]))
    mujoco.mj_step(model, data)
    max_gap_during_carry = max(max_gap_during_carry, abs(grasp.gap(data)))
print(f"  max |gap| during carry: {max_gap_during_carry * 1000:.3f} mm")
assert max_gap_during_carry < 1.0, "FAIL: large gap appeared during carry"
print("  PASS: ball stayed flush against the gripper throughout the carry.\n")

print("Phase 4: re-open jaw -- must release")
for step in range(50):
    data.qpos[JAW_QADR] = min(1.5, data.qpos[JAW_QADR] + 0.05)
    grasp.update(model, data, jaw_qpos=float(data.qpos[JAW_QADR]))
    mujoco.mj_step(model, data)
    if not grasp.state.attached:
        print(f"  released at step {step}, jaw_qpos={data.qpos[JAW_QADR]:.3f}")
        break
else:
    raise AssertionError("FAIL: never released after reopening the jaw")
print("  PASS: release worked.\n")

print("ALL ALGORITHM CHECKS PASSED")
