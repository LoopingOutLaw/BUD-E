"""Wrapper around MuJoCo/MJX for BUD-E's SO-101 5+1 DOF arm.

Scene built from the ggand0/pick-101 proven working model (so101_new_calib.xml
with finger pads, fingertip sites, gripperframe/graspframe sites). We load the
upstream MJCF and programmatically compose it with our pick-scene elements
(cube, target zone, cameras) using MuJoCo's MjSpec API.

CRITICAL PHYSICS SETTINGS (matching ggand0/pick-101 exactly):
  - Elliptic cone + noslip_iterations=3 + impratio=10
  - solref="0.001 1" solimp="0.99 0.99 0.001" (very stiff contacts)
  - znear=0.005 (prevents clipping through small finger meshes)
  - 3cm cube, mass=0.03, wood friction

The gripper has a single asymmetric moving jaw that rotates about its
hinge from closed (-0.175 rad) to open (+1.745 rad).

qpos layout (MuJoCo joint order — arm-first DFS):
    qpos[0:5]   = 5 arm revolutes
                  shoulder_pan, shoulder_lift, elbow_flex,
                  wrist_flex, wrist_roll
    qpos[5:6]   = gripper (single revolute, asymmetric jaw)
    qpos[6:13]  = cube freejoint (xyz + quat wxyz)

Action order: 5 arm + 1 gripper = 6D continuous.

COLLISION GROUPS
----------------
contype/conaffinity bitmasks:
    GROUP_DEFAULT (1) — robot, floor, table
    GROUP_BALL    (2) — cube
    GROUP_BOWL    (4) — target bowl

Cube gets contype=conaffinity=3, collides with everything.
Bowl gets contype=4, conaffinity=2 — collides with cube only, not robot.
"""
from __future__ import annotations

import math as _m
import os
from pathlib import Path

import mujoco
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from mujoco import mjx
    _MJX_AVAILABLE = True
except ImportError:
    _MJX_AVAILABLE = False

_ARM_SPEC_PATH = (
    Path(__file__).resolve().parents[3] / "urdf" / "so101_official" / "so101_new_calib.xml"
)

GROUP_DEFAULT = 1
GROUP_BALL = 2
GROUP_BOWL = 4

BALL_CONTYPE = GROUP_DEFAULT | GROUP_BALL       # 3
BALL_CONAFFINITY = GROUP_DEFAULT | GROUP_BALL   # 3
BOWL_CONTYPE = GROUP_BOWL                       # 4
BOWL_CONAFFINITY = GROUP_BALL                   # 2


def _add_ring_wall(parent_body, radius: float, wall_height: float, z_center: float,
                   n_segments: int, thickness: float, rgba, contype: int, conaffinity: int,
                   name_prefix: str, overlap: float = 1.18) -> None:
    """Build a circular wall out of `n_segments` thin boxes around `radius`."""
    circumference = 2 * _m.pi * radius
    seg_full_width = (circumference / n_segments) * overlap
    half_width = seg_full_width / 2.0
    for i in range(n_segments):
        ang = 2 * _m.pi * i / n_segments
        parent_body.add_geom(
            name=f"{name_prefix}_{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[thickness, half_width, wall_height / 2.0],
            pos=[radius * _m.cos(ang), radius * _m.sin(ang), z_center],
            euler=[0, 0, ang + _m.pi / 2],
            rgba=rgba,
            contype=contype,
            conaffinity=conaffinity,
            condim=3 if contype != 0 else 1,
        )


def _build_composite_spec() -> mujoco.MjSpec:
    """Build the composite pick-scene by extending the upstream arm spec.

    The upstream so101_new_calib.xml (from ggand0/pick-101) already includes:
      - finger pad collision geoms (static_finger_pad, moving_finger_pad)
      - fingertip sites (static_fingertip, moving_fingertip)
      - gripperframe site (TCP at fingertips, pos="0.0 0.0 -0.0981274")
      - graspframe site (midpoint between fingers)
      - wrist_cam camera
      - sts3215 position actuators with correct gains (kp=998.22 kv=2.731)
      - contype=1 conaffinity=1 on visual/collision geoms

    We add:
      - Physics options (elliptic cone, noslip, impratio, solref/solimp defaults)
      - Floor, table, cube, target zone, bowl
      - Additional fixed cameras
      - Sensors
      - znear visual setting
    """
    cwd = os.getcwd()
    arm_dir = _ARM_SPEC_PATH.parent
    os.chdir(arm_dir)
    try:
        spec = mujoco.MjSpec.from_file(_ARM_SPEC_PATH.name)

        # Physics settings matching ggand0/pick-101 exactly.
        # CRITICAL for reliable grasping — prevents cube sliding through fingers.
        spec.option.timestep = 0.002
        spec.option.iterations = 100
        spec.option.ls_iterations = 50
        spec.option.noslip_iterations = 3
        spec.option.impratio = 10
        spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

        # Default geom contact parameters — very stiff (matching pick-101).
        # solref=[0.001, 1]: extremely fast constraint stabilization
        # solimp=[0.99, 0.99, 0.001, 0, 0]: near-rigid impedance
        spec.default.geom.solref = np.array([0.001, 1.0])
        spec.default.geom.solimp = np.array([0.99, 0.99, 0.001, 0.0, 0.0])

        # Visual: znear=0.005 prevents clipping through small finger meshes
        spec.visual.map.znear = 0.005

        # Floor
        spec.worldbody.add_geom(
            name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[1, 1, 0.05], rgba=[0.7, 0.7, 0.8, 1], condim=3,
            friction=[1.0, 0.005, 0.0001],
        )

        # Table
        spec.worldbody.add_geom(
            name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, 0.25, 0.0], size=[0.35, 0.10, 0.02],
            rgba=[0.85, 0.78, 0.65, 1], condim=3,
        )

        # 3cm cube matching ggand0/pick-101 exactly:
        #   half-extent 0.015, mass 0.03, wood friction
        #   center at z=0.015 (floor + half_extent)
        ball = spec.worldbody.add_body(name="cube", pos=[0.25, 0.0, 0.015])
        ball.add_joint(name="cube_joint", type=mujoco.mjtJoint.mjJNT_FREE)
        ball.add_geom(
            name="cube_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.015, 0.015, 0.015],
            rgba=[0.9, 0.1, 0.1, 1],
            mass=0.03,
            friction=[0.5, 0.05, 0.001],
            contype=BALL_CONTYPE,
            conaffinity=BALL_CONAFFINITY,
        )

        # Target zone + bowl at (0.32, 0.16)
        tgt = spec.worldbody.add_body(name="target_zone", pos=[0.32, 0.16, 0.017])
        tgt.add_geom(
            name="target_zone_disc", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.06, 0.06, 0.002], rgba=[0.1, 0.3, 0.95, 1],
            contype=0, conaffinity=0,
        )
        tgt.add_geom(
            name="target_zone_inner", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.025, 0.025, 0.003], rgba=[0.95, 0.95, 1.0, 1],
            pos=[0, 0, 0.0005], contype=0, conaffinity=0,
        )

        bowl = spec.worldbody.add_body(name="bowl", pos=[0.32, 0.16, 0.017])
        bowl.add_geom(
            name="bowl_floor",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.032, 0.002],
            rgba=[0.30, 0.30, 0.36, 1],
            contype=BOWL_CONTYPE,
            conaffinity=BOWL_CONAFFINITY,
            condim=6,
            friction=[5.0, 0.5, 0.1],
        )
        _add_ring_wall(
            bowl, radius=0.038, wall_height=0.040, z_center=0.020,
            n_segments=24, thickness=0.003, rgba=[0.25, 0.25, 0.30, 1],
            contype=BOWL_CONTYPE, conaffinity=BOWL_CONAFFINITY,
            name_prefix="bowl_rim",
        )

        # Fixed world-frame cameras
        for nm, p, xy, f in [
            ("over_shoulder", [-0.05,  0.55, 0.50], [-0.7071, -0.7071, 0,  0.4814, -0.4814,  0.7325], 48),
            ("pov",           [ 0.40, -0.30, 0.65], [ 0.9806,  0.1961, 0, -0.1505,  0.7524,  0.6413], 55),
            ("front_top",     [ 0.30,  0.00, 0.80], [ 1.0000,  0.0000, 0,  0.0000,  0.9671,  0.2545], 42),
            ("portfolio",     [-0.20, -1.00, 0.85], [ 0.9231, -0.3846, 0,  0.2034,  0.4881,  0.8487], 36),
        ]:
            spec.worldbody.add_camera(name=nm, pos=p, xyaxes=xy, fovy=f)

        # Sensors matching pick-101
        spec.add_sensor(
            name="cube_pos",
            type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            objname="cube",
        )
        spec.add_sensor(
            name="gripper_pos",
            type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname="gripperframe",
        )
        spec.add_sensor(
            name="grasp_pos",
            type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname="graspframe",
        )

        return spec.compile()
    finally:
        os.chdir(cwd)


def load_arm_model(xml_path: str | Path | None = None) -> mujoco.MjModel:
    """Load the composite SO-101 pick scene as an MjModel."""
    if xml_path is not None:
        path = Path(xml_path)
        if not path.exists():
            raise FileNotFoundError(f"Arm scene not found at {path}")
        return mujoco.MjModel.from_xml_path(str(path))
    return _build_composite_spec()


def default_joint_angles(model: mujoco.MjModel) -> np.ndarray:
    """Home config for the 5 arm joints."""
    return np.asarray([0.0, -0.5, 0.95, -0.55, 0.0])


# qpos layout:
ARM_QPOS_START = 0
ARM_QPOS_END = 5
GRIPPER_QPOS_START = 5
GRIPPER_QPOS_END = 6
CUBE_QPOS_START = 6
CUBE_QPOS_END = 13

CUBE_REST_Z = 0.015    # 3cm cube center on floor (floor + half_extent=0.015)

N_ARM_JOINTS = 5
N_GRIPPER_JOINTS = 1
TOTAL_JOINT_DIM = N_ARM_JOINTS + N_GRIPPER_JOINTS  # 6


class SO101MJMJX:
    """JAX/jnp interface to the SO-101 pick scene."""

    def __init__(self, xml_path: str | Path | None = None):
        if not _MJX_AVAILABLE:
            raise ImportError(
                "SO101MJMJX requires jax and mujoco.mjx. "
                "Install with: pip install jax jaxlib mujoco-mjx"
            )
        self.model_mj = load_arm_model(xml_path)
        self.model = mjx.put_model(self.model_mj)
        self.n_arm = N_ARM_JOINTS
        self.nu = self.model_mj.nu
        self.action_dim = self.nu
        self.n_qpos = self.model_mj.nq
        self.n_qvel = self.model_mj.nv

    def make_data(self, joint_angles: np.ndarray | None = None,
                  cube_xyz: tuple[float, float, float] = (0.25, 0.0, 0.015)
                  ) -> mjx.Data:
        d = mjx.make_data(self.model)
        if joint_angles is not None:
            angles = jnp.asarray(joint_angles, dtype=jnp.float32)
            if angles.shape[0] < self.n_arm:
                pad = jnp.zeros(self.n_arm - angles.shape[0], dtype=jnp.float32)
                angles = jnp.concatenate([angles, pad])
            angles = angles[: self.n_arm]
            d = d.replace(qpos=d.qpos.at[ARM_QPOS_START:ARM_QPOS_END].set(angles))
        cube = jnp.asarray(cube_xyz, dtype=jnp.float32)
        d = d.replace(qpos=d.qpos.at[CUBE_QPOS_START:CUBE_QPOS_START + 3].set(cube))
        d = d.replace(qpos=d.qpos.at[CUBE_QPOS_START + 3:CUBE_QPOS_END].set(
            jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)))
        return d

    def reset(self, joint_angles: np.ndarray | None = None,
              cube_xyz: tuple[float, float, float] = (0.25, 0.0, 0.015)
              ) -> mjx.Data:
        return self.make_data(joint_angles, cube_xyz)

    @staticmethod
    def _to_action(action) -> jnp.ndarray:
        a = jnp.asarray(action, dtype=jnp.float32)
        if a.ndim == 1:
            a = a[None, :]
        return a

    def step_static(self, state: mjx.Data, action) -> mjx.Data:
        a = self._to_action(action)
        if a.ndim != 2:
            raise ValueError(f"action must be 1D or 2D, got shape {a.shape}")
        if a.shape[-1] != self.model_mj.nu:
            raise ValueError(
                f"Action last-dim {a.shape[-1]} != model.nu {self.model_mj.nu}.")
        is_batched = state.qpos.ndim > 1
        if not is_batched:
            new = state.replace(ctrl=a[0])
            return mjx.step(self.model, new)
        if a.shape[0] != state.qpos.shape[0]:
            raise ValueError(
                f"action batch {a.shape[0]} != state batch {state.qpos.shape[0]}")

        @jax.vmap
        def _vmap_step(d, act):
            d = d.replace(ctrl=act)
            return mjx.step(self.model, d)

        return _vmap_step(state, a)

    def jitted_step(self):
        return jax.jit(lambda state, action: self.step_static(state, action))

    def render(self, state: mjx.Data, height: int = 224, width: int = 224) -> np.ndarray:
        d = mjx.get_data(self.model_mj, state)
        try:
            if not hasattr(self, "_renderer") or self._renderer is None:
                self._renderer = mujoco.Renderer(self.model_mj, height=height, width=width)
            else:
                self._renderer._height = height
                self._renderer._width = width
            self._renderer.update_scene(d)
            return self._renderer.render()
        except Exception:
            return np.zeros((height, width, 3), dtype=np.uint8)

    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.asarray(self.model_mj.actuator_ctrlrange[:, 0]),
                np.asarray(self.model_mj.actuator_ctrlrange[:, 1]))
