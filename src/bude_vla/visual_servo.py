"""Camera-only planar object localization for the pick controller.

The runtime tracker never reads MuJoCo object state. In simulation, calibration
uses a grid of known marker placements, equivalent to calibrating a fixed real
camera with a checkerboard or a marker moved to measured table coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from bude_vla.envs.so101_mjx import (
    CUBE_QPOS_END,
    CUBE_QPOS_START,
    CUBE_REST_Z,
)
from bude_vla.perception import detect_red_candidates, detect_red_centroid


@dataclass(frozen=True)
class PlanarHomography:
    """Project normalized image coordinates onto a fixed world plane."""

    matrix: np.ndarray

    @classmethod
    def fit(cls, image_xy: np.ndarray, world_xy: np.ndarray) -> "PlanarHomography":
        image_xy = np.asarray(image_xy, dtype=np.float64)
        world_xy = np.asarray(world_xy, dtype=np.float64)
        if image_xy.shape != world_xy.shape or image_xy.ndim != 2 or image_xy.shape[1] != 2:
            raise ValueError("image_xy and world_xy must both have shape (N, 2)")
        if image_xy.shape[0] < 4:
            raise ValueError("at least four calibration correspondences are required")

        rows = []
        values = []
        for (u, v), (x, y) in zip(image_xy, world_xy, strict=True):
            rows.append([u, v, 1.0, 0.0, 0.0, 0.0, -x * u, -x * v])
            values.append(x)
            rows.append([0.0, 0.0, 0.0, u, v, 1.0, -y * u, -y * v])
            values.append(y)
        params, _residuals, rank, _singular = np.linalg.lstsq(
            np.asarray(rows, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
            rcond=None,
        )
        if rank < 8:
            raise ValueError("degenerate planar calibration points")
        matrix = np.asarray([
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
            [params[6], params[7], 1.0],
        ])
        return cls(matrix=matrix)

    def image_to_world(self, image_xy: np.ndarray) -> np.ndarray:
        points = np.asarray(image_xy, dtype=np.float64)
        one = points.ndim == 1
        points = np.atleast_2d(points)
        if points.shape[1] != 2:
            raise ValueError("image_xy must have shape (2,) or (N, 2)")
        homogeneous = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1
        )
        projected = homogeneous @ self.matrix.T
        denom = projected[:, 2:3]
        if np.any(np.abs(denom) < 1e-9):
            raise ValueError("calibration projected a point to infinity")
        result = projected[:, :2] / denom
        return result[0] if one else result

    def errors(self, image_xy: np.ndarray, world_xy: np.ndarray) -> np.ndarray:
        predicted = self.image_to_world(image_xy)
        return np.linalg.norm(predicted - np.asarray(world_xy), axis=1)


def _place_calibration_cube(data: mujoco.MjData, x: float, y: float) -> None:
    data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = [x, y, CUBE_REST_Z]
    data.qpos[CUBE_QPOS_START + 3:CUBE_QPOS_START + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[CUBE_QPOS_START:CUBE_QPOS_END] = 0.0
    mujoco.mj_forward(data.model, data)


def calibrate_red_cube_homography(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    camera_id: int,
    x_range: tuple[float, float] = (0.15, 0.35),
    y_range: tuple[float, float] = (-0.10, 0.10),
    grid_size: int = 5,
) -> tuple[PlanarHomography, dict[str, float]]:
    """Calibrate image-to-table mapping from known marker placements.

    This function is simulator-specific calibration tooling. The returned
    homography is the only calibration artifact used by the runtime tracker.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    saved_ctrl = data.ctrl.copy()
    image_points: list[np.ndarray] = []
    world_points: list[np.ndarray] = []
    try:
        for x in np.linspace(*x_range, grid_size):
            for y in np.linspace(*y_range, grid_size):
                _place_calibration_cube(data, float(x), float(y))
                renderer.update_scene(data, camera=camera_id)
                image = np.asarray(renderer.render()).copy()
                detection = detect_red_centroid(image)
                if detection[2] <= 0.5:
                    continue
                image_points.append(detection[:2].astype(np.float64))
                world_points.append(np.asarray([x, y], dtype=np.float64))
    finally:
        data.qpos[:] = saved_qpos
        data.qvel[:] = saved_qvel
        data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(model, data)

    if len(image_points) < 8:
        raise RuntimeError(
            f"red cube detected at only {len(image_points)} calibration points"
        )
    image_array = np.stack(image_points)
    world_array = np.stack(world_points)
    homography = PlanarHomography.fit(image_array, world_array)
    errors = homography.errors(image_array, world_array)
    stats = {
        "points": float(len(errors)),
        "mean_error_m": float(errors.mean()),
        "p95_error_m": float(np.percentile(errors, 95)),
        "max_error_m": float(errors.max()),
    }
    return homography, stats


class RedCubeWorldTracker:
    """Track cube XY from top-camera RGB and expose a world XYZ provider."""

    def __init__(
        self,
        renderer: mujoco.Renderer,
        camera_id: int,
        homography: PlanarHomography,
        workspace_x: tuple[float, float] = (0.13, 0.52),
        workspace_y: tuple[float, float] = (-0.13, 0.20),
        smoothing: float = 0.25,
        max_jump_m: float = 0.025,
        occlusion_radius_m: float = 0.055,
    ):
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        if max_jump_m <= 0.0:
            raise ValueError("max_jump_m must be positive")
        self.renderer = renderer
        self.camera_id = camera_id
        self.homography = homography
        self.workspace_x = workspace_x
        self.workspace_y = workspace_y
        self.smoothing = float(smoothing)
        self.max_jump_m = float(max_jump_m)
        self.occlusion_radius_m = float(occlusion_radius_m)
        self._last_xy: np.ndarray | None = None
        self._last_time: float | None = None
        self._last_detection = np.zeros(3, dtype=np.float32)
        self._occlusion_site_id: int | None = None

    @property
    def last_detection(self) -> np.ndarray:
        return self._last_detection.copy()

    @property
    def last_xy(self) -> np.ndarray | None:
        return None if self._last_xy is None else self._last_xy.copy()

    def reset(self) -> None:
        self._last_xy = None
        self._last_time = None
        self._last_detection = np.zeros(3, dtype=np.float32)

    def _gripper_occludes_estimate(self, data: mujoco.MjData) -> bool:
        if self._last_xy is None or self.occlusion_radius_m <= 0.0:
            return False
        if self._occlusion_site_id is None:
            self._occlusion_site_id = mujoco.mj_name2id(
                data.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
            )
        if self._occlusion_site_id < 0:
            return False
        estimate_xyz = np.asarray(
            [self._last_xy[0], self._last_xy[1], CUBE_REST_Z],
            dtype=np.float64,
        )
        distance = np.linalg.norm(
            data.site_xpos[self._occlusion_site_id] - estimate_xyz
        )
        return bool(distance < self.occlusion_radius_m)

    def update(self, data: mujoco.MjData, force: bool = False) -> np.ndarray | None:
        now = float(data.time)
        if not force and self._last_time == now:
            return self.last_xy
        self._last_time = now
        if not force and self._gripper_occludes_estimate(data):
            return self.last_xy

        self.renderer.update_scene(data, camera=self.camera_id)
        image = np.asarray(self.renderer.render()).copy()
        detections = detect_red_candidates(image)
        if not len(detections):
            self._last_detection = np.zeros(3, dtype=np.float32)
            return self.last_xy

        world_candidates = self.homography.image_to_world(detections[:, :2])
        valid = (
            (world_candidates[:, 0] >= self.workspace_x[0])
            & (world_candidates[:, 0] <= self.workspace_x[1])
            & (world_candidates[:, 1] >= self.workspace_y[0])
            & (world_candidates[:, 1] <= self.workspace_y[1])
        )
        valid_indices = np.flatnonzero(valid)
        if not len(valid_indices):
            return self.last_xy

        if self._last_xy is None:
            selected = int(valid_indices[0])
        else:
            distances = np.linalg.norm(
                world_candidates[valid_indices] - self._last_xy[None, :], axis=1
            )
            nearest_local = int(np.argmin(distances))
            if distances[nearest_local] > self.max_jump_m:
                return self.last_xy
            selected = int(valid_indices[nearest_local])

        candidate = world_candidates[selected]
        self._last_detection = np.asarray(
            [detections[selected, 0], detections[selected, 1], 1.0],
            dtype=np.float32,
        )
        if self._last_xy is None:
            self._last_xy = candidate
        else:
            self._last_xy = (
                self.smoothing * self._last_xy + (1.0 - self.smoothing) * candidate
            )
        return self.last_xy

    def position(self, data: mujoco.MjData) -> np.ndarray:
        """Return the locked estimate; initialize it from RGB when needed."""
        xy = self.last_xy
        if xy is None:
            xy = self.update(data, force=True)
        if xy is None:
            raise RuntimeError("red cube is not visible and no previous estimate exists")
        return np.asarray([xy[0], xy[1], CUBE_REST_Z], dtype=np.float64)

    def __call__(self, data: mujoco.MjData) -> np.ndarray:
        return self.position(data)

    def reacquire(self, data: mujoco.MjData) -> np.ndarray | None:
        """Refresh the estimate after the controller has backed away."""
        return self.update(data, force=True)
