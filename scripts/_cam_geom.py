"""Place cameras so they ACTUALLY see the cube workspace.

Replaces the corner-of-table-cam positions that ship in ur5e_scene.xml
with positions derived from the actual workspace geometry:
  - table center at (0.5, 0, 0.4)
  - cube working region: x ∈ [0.50, 0.75], y ∈ [-0.15, 0.15], z = 0.445
  - target_zone at (0.85, 0, 0.42)
"""
from __future__ import annotations
import mujoco
import numpy as np

XML = '/home/aditya/bude_vla/urdf/ur5e_scene.xml'

# Cube working region center
CUBE_REGION = (0.625, 0.0, 0.445)
CUBE_REGION_HALF = (0.20, 0.20)  # x and y half-extents
TARGET = (0.85, 0.0, 0.42)
# Region bounding box corners in world:
#   x: 0.425 - 0.85  (cube workspace left to target right)
#   y: -0.2  - 0.2   (cube x range)
#   z: 0.42 - 0.49   (table top to above-cube)


def find_best_cam_pos(cube_pos, cam_height=1.0, fovy_deg=42, margin=1.2,
                      fwd_axis=(0, 0, -1)):
    """Return (cam_pos, cam_quat) such that cam looks down at cube_pos.

    cam_height above scene, fovy chosen to fit cube with given margin.
    fwd_axis is the camera's looking direction in WORLD frame.
    """
    cam_pos = np.array([cube_pos[0], cube_pos[1], cam_pos_height_helper(cube_pos, cam_height)])
    cam_pos = np.array([cube_pos[0], cube_pos[1], cam_height])
    # Look-at quaternion: MuJoCo's default cam looks down -z in PARENT (world) frame
    # So at cam_pos looking at cube_pos, by default "looking dir" is world -z.
    # We need explicit quat if cube is not directly below.
    # For straight-down: keep fwd = (0,0,-1) but cam_pos needs to be ABOVE cube
    if fwd_axis == (0, 0, -1):
        cam_pos = np.array([cube_pos[0], cube_pos[1], cam_height])
        # Default cam looking down -z — works for top-down
        return cam_pos, np.array([1.0, 0.0, 0.0, 0.0])
    raise NotImplementedError("only straight-down supported")


def cam_pos_height_helper(cube_pos, cam_height):
    return cam_height


def main():
    print(f'Cube region: {CUBE_REGION}')
    print(f'Target: {TARGET}')

    # Best overhead cam: directly above table center
    overhead = np.array([0.65, 0.0, 1.0])
    overhead_quat = np.array([1.0, 0.0, 0.0, 0.0])  # default forward = -z

    # Workspace-bounds test
    print(f'\n[overhead]  pos={overhead}, looking straight down')
    print(f'   workspace extent visible at z=0.445: w=2*{overhead[2]-0.445}*tan(21°)='
          f'{2*(overhead[2]-0.445)*np.tan(np.deg2rad(21)):.3f}m  '
          f'(need ~0.4m x, 0.3m y)')

    # print current cam config in XML
    m = mujoco.MjModel.from_xml_path(XML)
    for i in range(m.ncam):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i)
        print(f'  [{i}] {name:18s} pos={m.cam_pos[i]} fovy={m.cam_fovy[i]}°')


if __name__ == '__main__':
    main()
