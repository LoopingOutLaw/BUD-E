"""Wrapper around MuJoCo/MJX for BUD-E's SO-101 5+1 DOF arm.

The arm comes from the official LeRobot/SO-ARM100 hardware project
(TheRobotStudio/SO-ARM100 on GitHub, Simulation/SO101). We load the
upstream pre-converted MJCF (so101_official/so101_new_calib.xml) and
programmatically compose it with our pick-scene elements (cube, table,
cameras, lights) using MuJoCo's MjSpec API.

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
contype/conaffinity bitmasks used throughout this scene:
    GROUP_DEFAULT (1) — robot, floor, table, pedestal (engine default,
                         unspecified geoms get contype=conaffinity=1)
    GROUP_BALL    (2) — the pick-and-place ball
    GROUP_BOWL    (4) — both bowls (pick bowl + target bowl)

The ball gets contype=conaffinity=GROUP_DEFAULT|GROUP_BALL (3), so it
keeps colliding with the robot/floor/table as before. Both bowls get
contype=GROUP_BOWL, conaffinity=GROUP_BALL — they collide with the ball
ONLY, never the robot. This fixes two bugs in the previous version:
  - the target bowl used contype=0/conaffinity=0 ("never collide with
    anything"), which meant it was purely decorative and could not
    physically catch/contain a dropped ball at all;
  - the pick bowl used the engine default (collides with everything),
    which meant it could physically obstruct the gripper reaching in.
"""
from __future__ import annotations

import math as _m
import os
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

_ARM_SPEC_PATH = (
    Path(__file__).resolve().parents[3] / "urdf" / "so101_official" / "so101_new_calib.xml"
)

GROUP_DEFAULT = 1
GROUP_BALL = 2
GROUP_BOWL = 4

BALL_CONTYPE = GROUP_DEFAULT | GROUP_BALL       # 3 — collides with robot/floor/table AND bowls
BALL_CONAFFINITY = GROUP_DEFAULT | GROUP_BALL   # 3
BOWL_CONTYPE = GROUP_BOWL                       # 4 — collides with...
BOWL_CONAFFINITY = GROUP_BALL                   # ...the ball only, never the robot


def _add_ring_wall(parent_body, radius: float, wall_height: float, z_center: float,
                   n_segments: int, thickness: float, rgba, contype: int, conaffinity: int,
                   name_prefix: str, overlap: float = 1.18) -> None:
    """Build a circular wall out of `n_segments` thin boxes around `radius`.

    `overlap` widens each segment's tangential extent past its even share
    of the circumference (1.0 = exactly touching, no gaps; >1.0 = slight
    overlap) so the wall reads as a smooth ring rather than a slatted
    fence with visible gaps between segments. The previous bowls used 12
    segments sized to barely cover ~70% of the circumference, leaving
    ~3-4mm gaps — small enough that the 25mm ball couldn't squeeze
    through, but visually choppy and an unnecessary source of debris
    leak risk for anything smaller.
    """
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

    chdir to the arm dir for the whole build so that mesh asset paths in the
    upstream MJCF (`<compiler meshdir="assets"/>`) and the resulting MjSpec
    both resolve to the actual STL folder.
    """
    cwd = os.getcwd()
    arm_dir = _ARM_SPEC_PATH.parent
    os.chdir(arm_dir)
    try:
        spec = mujoco.MjSpec.from_file(_ARM_SPEC_PATH.name)

        spec.worldbody.add_geom(
            name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[1, 1, 0.05], rgba=[0.7, 0.7, 0.8, 1], condim=3,
        )
        spec.worldbody.add_geom(
            name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, 0.25, 0.0], size=[0.35, 0.10, 0.02],
            rgba=[0.85, 0.78, 0.65, 1], condim=3,
        )

        # Pick bowl: constrains ball laterally during grasp. Collides with
        # the ball ONLY (GROUP_BOWL / GROUP_BALL) so it can't obstruct the
        # gripper reaching in -- previously it used the engine default,
        # which meant it physically collided with the robot too.
        pick_bowl = spec.worldbody.add_body(name="pick_bowl", pos=[0.30, 0.0, 0.016])
        pick_bowl.add_geom(
            name="pick_bowl_floor",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.022, 0.001],
            rgba=[0.45, 0.45, 0.50, 1],
            contype=BOWL_CONTYPE,
            conaffinity=BOWL_CONAFFINITY,
            condim=3,
        )
        _add_ring_wall(
            pick_bowl, radius=0.026, wall_height=0.016, z_center=0.008,
            n_segments=20, thickness=0.0015, rgba=[0.40, 0.40, 0.45, 1],
            contype=BOWL_CONTYPE, conaffinity=BOWL_CONAFFINITY,
            name_prefix="pick_bowl_rim",
        )

        # 25 mm-diameter ball (50 mm cube won't fit between the closed jaws).
        # Body remains named "cube" to avoid sweeping ik.py, recorder, and
        # recorded .npz schema; treat "cube" as the historical alias for the
        # pick payload.
        # contype/conaffinity explicitly set (rather than relying on the
        # engine default) so it's clear this is the one body that needs to
        # collide with BOTH the robot/floor/table group AND the bowl group.
        ball = spec.worldbody.add_body(name="cube", pos=[0.30, 0.0, 0.030])
        ball.add_joint(name="cube_free", type=mujoco.mjtJoint.mjJNT_FREE)
        ball.add_geom(
            name="cube_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.0125],
            rgba=[0.85, 0.05, 0.05, 1],
            mass=0.05,
            condim=4,
            solref=[0.02, 1.0],
            solimp=[0.9, 0.95, 0.001, 0.5, 2],
            contype=BALL_CONTYPE,
            conaffinity=BALL_CONAFFINITY,
        )

        tgt = spec.worldbody.add_body(name="target_zone", pos=[0.30, 0.40, 0.021])
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
        # Bowl to keep the ball from rolling off after drop.
        # FIXED: previously contype=0/conaffinity=0 ("never collide with
        # anything"), which made this bowl purely decorative -- a dropped
        # ball would roll straight through the "walls" with no physical
        # containment at all. Now it uses the same bowl/ball-only group as
        # the pick bowl: it collides with the ball, never the robot, so it
        # actually catches and holds the dropped ball while staying out of
        # the arm's way.
        bowl = spec.worldbody.add_body(name="bowl", pos=[0.30, 0.40, 0.016])
        bowl.add_geom(
            name="bowl_floor",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.028, 0.001],
            rgba=[0.30, 0.30, 0.36, 1],
            contype=BOWL_CONTYPE,
            conaffinity=BOWL_CONAFFINITY,
            condim=3,
        )
        _add_ring_wall(
            bowl, radius=0.033, wall_height=0.020, z_center=0.010,
            n_segments=20, thickness=0.0015, rgba=[0.25, 0.25, 0.30, 1],
            contype=BOWL_CONTYPE, conaffinity=BOWL_CONAFFINITY,
            name_prefix="bowl_rim",
        )

        # xyaxes = [X_cam, Y_cam] in the *parent body* frame, computed as:
        #   fwd   = normalize(look_at - pos)
        #   X_cam = normalize(cross(fwd, world_up))   # right in image
        #   Y_cam = normalize(cross(X_cam, fwd))      # up in image
        #
        # These four are fixed world-frame cameras (parent = worldbody).
        for nm, p, xy, f in [
            ("over_shoulder", [-0.05,  0.55, 0.50], [-0.7071, -0.7071, 0,  0.4814, -0.4814,  0.7325], 48),
            ("pov",           [ 0.40, -0.30, 0.65], [ 0.9806,  0.1961, 0, -0.1505,  0.7524,  0.6413], 55),
            ("front_top",     [ 0.30,  0.00, 0.80], [ 1.0000,  0.0000, 0,  0.0000,  0.9671,  0.2545], 42),
            ("portfolio",     [-0.20, -1.00, 0.85], [ 0.9231, -0.3846, 0,  0.2034,  0.4881,  0.8487], 36),
        ]:
            spec.worldbody.add_camera(name=nm, pos=p, xyaxes=xy, fovy=f)

        # ------------------------------------------------------------------
        # Eye-in-hand (wrist) camera
        # ------------------------------------------------------------------
        # The camera MUST be a child of the end-effector body so it moves
        # with the arm.  Parenting it to worldbody (as the other cameras
        # above) produces a fixed world-frame view that never tracks the arm.
        #
        # SO-101 / SO-ARM100 body chain (DFS order from so101_new_calib.xml):
        #   base → Shoulder_Pan → Shoulder_Lift → Upper_Arm
        #        → Wrist_Pitch  → Wrist_Roll    → Fixed_Jaw  ← gripper base
        #                                          └ Moving_Jaw (gripper hinge)
        #
        # "Fixed_Jaw" is the rigid gripper body that holds both fingers; the
        # "gripper" actuated joint rotates Moving_Jaw relative to it.
        # Attaching the camera here gives a view that follows every DOF.
        #
        # If the name has changed, run this once to discover all body names:
        #   m = load_arm_model()
        #   for i in range(m.nbody):
        #       print(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i))
        # Upstream SO-101 body chain (so101_new_calib.xml):
        #   base → shoulder → upper_arm → lower_arm → wrist → gripper → moving_jaw
        # "gripper" is the rigid body whose child (moving_jaw_so101_v1) hinges
        # about the actuated `gripper` joint — i.e. the EE.  No body literally
        # named "Fixed_Jaw" exists in this MJCF (that name is from the LeRobot
        # URDF, not the MJCF).
        _EE_BODY = "gripper"
        ee_body = next((b for b in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY)
                        if b.name == _EE_BODY), None)
        if ee_body is None:
            raise RuntimeError(
                f"End-effector body '{_EE_BODY}' not found in SO-101 spec. "
                "Run load_arm_model() and enumerate m.nbody to find the correct name."
            )

        # Local-frame placement on the EE (gripper) body.
        #   +Y_body  = toward fingertips / approach / workspace
        #   -Y_body  = toward palm back / forearm (away from workspace)
        #   +Z_body  = "up" lateral axis when the arm is in home pose
        #
        # VLA wrist-cam design goal:
        #   workspace (ball + table) fills center frame; gripper fingertips
        #   just visible at the top edge — matching eye-in-hand POV in VLA papers.
        #
        # pos:  14 cm behind palm (-Y), 11 cm above (+Z)  → well clear of geometry
        # tilt: 50 degrees below +Y_body horizontal
        #   X_cam = [1,  0,      0    ]
        #   Y_cam = [0,  0.766,  0.643]  (camera up = sin50, cos50)
        #   look  = [0,  0.643, -0.766]  (forward + steeply down)
        ee_body.add_camera(
            name="wrist",
            pos=[0.0, -0.14, 0.11],
            xyaxes=[0, 0.766, 0.643,  -1, 0, 0],
            fovy=90,
        )

        # Site on the moving-jaw body at the finger-tip contact surface.
        jaw_body = next(
            (b for b in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY)
             if b.name == "moving_jaw_so101_v1"), None,
        )
        if jaw_body is not None:
            jaw_body.add_site(
                name="jaw_contact",
                pos=[-0.001, -0.025, 0.019],
                size=0.005,
                rgba=[1, 0.3, 0.3, 0.6],
            )

        spec.worldbody.add_site(
            name="workspace_origin", pos=[0.30, 0.20, 0.04],
            size=0.01, rgba=[0, 1, 0, 0.5],
        )

        return spec.compile()
    finally:
        os.chdir(cwd)


def load_arm_model(xml_path: str | Path | None = None) -> mujoco.MjModel:
    """Load the composite SO-101 pick scene as an MjModel.

    If xml_path is given, load that file instead (for testing/debug).
    Otherwise build the composite programmatically via MjSpec.
    """
    if xml_path is not None:
        path = Path(xml_path)
        if not path.exists():
            raise FileNotFoundError(f"Arm scene not found at {path}")
        return mujoco.MjModel.from_xml_path(str(path))
    return _build_composite_spec()


def default_joint_angles(model: mujoco.MjModel) -> np.ndarray:
    """Home config for the 5 arm joints (in joint-ID order)."""
    return np.asarray([0.0, -0.5, 0.95, -0.55, 0.0])


# qpos layout — MuJoCo arm-first DFS order:
#   qpos[0:5]   = 5 arm revolutes (shoulder_pan … wrist_roll)
#   qpos[5:6]   = gripper (single revolute)
#   qpos[6:13]  = cube freejoint (xyz + quat wxyz)
ARM_QPOS_START = 0
ARM_QPOS_END = 5
GRIPPER_QPOS_START = 5
GRIPPER_QPOS_END = 6
CUBE_QPOS_START = 6
CUBE_QPOS_END = 13

N_ARM_JOINTS = 5
N_GRIPPER_JOINTS = 1
TOTAL_JOINT_DIM = N_ARM_JOINTS + N_GRIPPER_JOINTS  # 6


class SO101MJMJX:
    """JAX/jnp interface to the SO-101 pick scene."""

    def __init__(self, xml_path: str | Path | None = None):
        self.model_mj = load_arm_model(xml_path)
        self.model = mjx.put_model(self.model_mj)
        self.n_arm = N_ARM_JOINTS
        self.nu = self.model_mj.nu
        self.action_dim = self.nu
        self.n_qpos = self.model_mj.nq
        self.n_qvel = self.model_mj.nv

    def make_data(self, joint_angles: np.ndarray | None = None,
                  cube_xyz: tuple[float, float, float] = (0.30, 0.0, 0.030)
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
              cube_xyz: tuple[float, float, float] = (0.30, 0.0, 0.030)
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
