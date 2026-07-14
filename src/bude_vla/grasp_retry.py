"""Robot-side closed-loop grasp verification and local retry.

The controller intentionally uses only signals available on a physical arm:
the policy action, measured gripper joint position, measured TCP position, and
an optional target reconstructed from calibrated RGB. Simulator contacts and
cube coordinates are never inputs to this state machine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def add_local_grasp_retry_args(parser: Any) -> None:
    """Add shared local-retry deployment flags to an argparse parser."""
    group = parser.add_argument_group("local closed-loop grasp retry")
    group.add_argument(
        "--local-grasp-retry",
        action="store_true",
        help=("Verify grasp from measured jaw aperture before transport; open, "
              "back off, and visually replan locally after an empty close."),
    )
    group.add_argument("--local-grasp-retries", type=int, default=2)
    group.add_argument("--grasp-close-trigger", type=float, default=0.0)
    group.add_argument(
        "--grasp-qpos-threshold",
        type=float,
        default=0.08,
        help=("Measured gripper position separating an obstructed grasp from an "
              "empty close. Calibrate this on physical hardware."),
    )
    group.add_argument(
        "--grasp-verify-steps",
        type=int,
        default=2,
        help="Policy-rate frames allowed for jaw feedback to settle after close.",
    )
    group.add_argument("--grasp-open-steps", type=int, default=5)
    group.add_argument("--grasp-backoff-steps", type=int, default=10)
    group.add_argument("--grasp-backoff-height", type=float, default=0.055)
    group.add_argument("--grasp-open-value", type=float, default=1.05)
    group.add_argument(
        "--local-grasp-recovery",
        choices=("rgb", "policy"),
        default="rgb",
        help=("Recovery after an empty close. 'rgb' uses calibrated camera "
              "localization for only the local re-grasp; 'policy' merely "
              "replans the VLA after backing off."),
    )


def local_grasp_retry_config_from_args(
    args: Any,
    *,
    ee_delta_scale: float = 0.05,
) -> "LocalGraspRetryConfig":
    """Build the shared controller config from parsed deployment flags."""
    return LocalGraspRetryConfig(
        max_retries=args.local_grasp_retries,
        close_command_threshold=args.grasp_close_trigger,
        grasp_qpos_threshold=args.grasp_qpos_threshold,
        verification_steps=args.grasp_verify_steps,
        open_steps=args.grasp_open_steps,
        backoff_steps=args.grasp_backoff_steps,
        backoff_height=args.grasp_backoff_height,
        open_gripper_value=args.grasp_open_value,
        recovery_mode=args.local_grasp_recovery,
        ee_delta_scale=ee_delta_scale,
    )


@dataclass(frozen=True)
class LocalGraspRetryConfig:
    """Timing and calibration for :class:`LocalGraspRetryController`.

    ``max_retries`` counts additional local attempts after the initial close.
    The default aperture threshold sits well between the SO-101 simulator's
    empty-close position (-0.175 rad) and its 30 mm cube grasp (>0.14 rad).
    Recalibrate that threshold from measured hardware before real deployment.
    """

    max_retries: int = 2
    close_command_threshold: float = 0.0
    grasp_qpos_threshold: float = 0.08
    verification_steps: int = 2
    open_steps: int = 5
    backoff_steps: int = 10
    backoff_height: float = 0.055
    open_gripper_value: float = 1.05
    closed_gripper_value: float = -0.8
    probe_gripper_value: float = -0.25
    visual_tighten_delta: float = 0.25
    recovery_mode: str = "rgb"
    visual_approach_z_offset: float = 0.040
    visual_grasp_z_offset: float = 0.010
    visual_lift_z_offset: float = 0.080
    visual_grasp_y_offset: float = -0.015
    visual_approach_steps: int = 40
    visual_descent_steps: int = 50
    visual_close_steps: int = 60
    visual_tighten_steps: int = 10
    visual_lift_steps: int = 30
    visual_settle_steps: int = 2
    visual_target_tolerance: float = 0.003
    ee_delta_scale: float = 0.05

    def __post_init__(self) -> None:
        integer_fields = (
            "max_retries",
            "verification_steps",
            "open_steps",
            "backoff_steps",
            "visual_approach_steps",
            "visual_descent_steps",
            "visual_close_steps",
            "visual_tighten_steps",
            "visual_lift_steps",
            "visual_settle_steps",
        )
        for name in integer_fields:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.backoff_height < 0:
            raise ValueError("backoff height must be non-negative")
        if self.recovery_mode not in {"rgb", "policy"}:
            raise ValueError("recovery_mode must be 'rgb' or 'policy'")
        if self.visual_target_tolerance < 0:
            raise ValueError("visual_target_tolerance must be non-negative")


@dataclass(frozen=True)
class GraspRetryStep:
    action: np.ndarray
    reset_policy: bool = False
    abort_attempt: bool = False
    event: str | None = None
    phase: str = "tracking"


class LocalGraspRetryController:
    """Gate transport on a verified grasp and recover locally after a miss.

    After a close request, the learned actions remain untouched for a short jaw
    settling window. Measured jaw position then distinguishes a cube-blocked
    aperture from an empty close. Verified trajectories remain bit-for-bit
    policy-controlled. Failed attempts open and back off vertically, then use
    calibrated RGB for a bounded local re-grasp before returning control to a
    freshly replanned VLA. No arm-home or cube reset occurs.
    """

    _SUPPORTED_ACTION_SPACES = {"ee_abs", "ee_delta"}

    def __init__(
        self,
        action_space: str,
        config: LocalGraspRetryConfig | None = None,
    ) -> None:
        self.action_space = str(action_space)
        if self.action_space not in self._SUPPORTED_ACTION_SPACES:
            raise ValueError(
                "local grasp retry requires ee_abs or ee_delta actions; "
                f"got {self.action_space!r}"
            )
        self.config = config or LocalGraspRetryConfig()
        self.reset()

    def reset(self) -> None:
        self.phase = "tracking"
        self.phase_step = 0
        self.retries_used = 0
        self.grasp_verified = False
        self._recovery_start_xyz: np.ndarray | None = None
        self._abort_xyz: np.ndarray | None = None
        self._disable_after_recovery = False
        self._visual_cube_xyz: np.ndarray | None = None
        self._visual_close_anchor_xyz: np.ndarray | None = None
        self._settled_steps = 0
        self._lost_grasp_steps = 0
        self._grasp_hold_value = self.config.closed_gripper_value
        self._grasp_contact_value = self.config.probe_gripper_value
        self._obstruction_steps = 0

    @property
    def active(self) -> bool:
        return self.phase not in {
            "tracking", "verified", "released", "disabled"
        }

    @property
    def blocks_policy(self) -> bool:
        """Whether execution should pause the learned action-chunk cursor."""
        return self.phase in {
            "opening",
            "backing_off",
            "visual_approach",
            "visual_descent",
            "visual_close",
            "visual_tighten",
            "visual_lift",
        }

    @property
    def needs_visual_reacquire(self) -> bool:
        """Whether the rollout should refresh RGB localization this frame."""
        return (
            self.config.recovery_mode == "rgb"
            and self.phase == "backing_off"
            and self.phase_step >= self.config.backoff_steps
            and not self._disable_after_recovery
        )

    @staticmethod
    def _valid_visual_target(target: np.ndarray | None) -> bool:
        if target is None:
            return False
        array = np.asarray(target)
        return array.shape == (3,) and bool(np.all(np.isfinite(array)))

    def _visual_target(self, *, approach: bool) -> np.ndarray:
        assert self._visual_cube_xyz is not None
        target = self._visual_cube_xyz.copy()
        target[1] += self.config.visual_grasp_y_offset
        target[2] += (
            self.config.visual_approach_z_offset
            if approach else self.config.visual_grasp_z_offset
        )
        return target

    def _target_is_settled(
        self, current_tcp_xyz: np.ndarray, target_xyz: np.ndarray
    ) -> bool:
        distance = np.linalg.norm(current_tcp_xyz - target_xyz)
        if distance <= self.config.visual_target_tolerance:
            self._settled_steps += 1
        else:
            self._settled_steps = 0
        return self._settled_steps >= self.config.visual_settle_steps

    def _target_action(
        self,
        policy_action: np.ndarray,
        current_tcp_xyz: np.ndarray,
        target_xyz: np.ndarray,
        gripper_value: float,
    ) -> np.ndarray:
        out = np.asarray(policy_action, dtype=np.float64).copy()
        current = np.asarray(current_tcp_xyz, dtype=np.float64)
        target = np.asarray(target_xyz, dtype=np.float64)
        if self.action_space == "ee_abs":
            out[:3] = target
        else:
            out[:3] = np.clip(
                target - current,
                -self.config.ee_delta_scale,
                self.config.ee_delta_scale,
            )
        out[-1] = gripper_value
        return out.astype(np.float32)

    def _override(
        self,
        policy_action: np.ndarray,
        current_tcp_xyz: np.ndarray,
        target_xyz: np.ndarray,
        gripper_value: float,
        *,
        reset_policy: bool = False,
        event: str | None = None,
    ) -> GraspRetryStep:
        return GraspRetryStep(
            action=self._target_action(
                policy_action, current_tcp_xyz, target_xyz, gripper_value
            ),
            reset_policy=reset_policy,
            event=event,
            phase=self.phase,
        )

    def _begin_recovery(
        self,
        policy_action: np.ndarray,
        current_tcp_xyz: np.ndarray,
        *,
        reason: str,
    ) -> GraspRetryStep:
        can_retry = self.retries_used < self.config.max_retries
        if can_retry:
            self.retries_used += 1
        self._disable_after_recovery = not can_retry
        self.grasp_verified = False
        self._obstruction_steps = 0
        self.phase = "opening"
        self.phase_step = 0
        self._recovery_start_xyz = np.asarray(
            current_tcp_xyz, dtype=np.float64
        ).copy()
        suffix = f"retry_{self.retries_used}" if can_retry else "exhausted"
        return self._override(
            policy_action,
            current_tcp_xyz,
            self._recovery_start_xyz,
            self.config.open_gripper_value,
            reset_policy=True,
            event=f"{reason}_{suffix}",
        )

    def step(
        self,
        policy_action: np.ndarray,
        current_tcp_xyz: np.ndarray,
        measured_gripper_qpos: float,
        visual_target_xyz: np.ndarray | None = None,
    ) -> GraspRetryStep:
        """Filter one policy action using current robot feedback."""
        action = np.asarray(policy_action, dtype=np.float32)
        tcp = np.asarray(current_tcp_xyz, dtype=np.float64)
        qpos = float(measured_gripper_qpos)

        if self.phase == "verified":
            if float(action[-1]) > self.config.close_command_threshold:
                self.phase = "released"
                self._lost_grasp_steps = 0
                return GraspRetryStep(action=action, phase=self.phase)
            if qpos < self.config.grasp_qpos_threshold:
                self._lost_grasp_steps += 1
                if self._lost_grasp_steps >= self.config.verification_steps:
                    return self._begin_recovery(
                        action, tcp, reason="grasp_lost"
                    )
            else:
                self._lost_grasp_steps = 0
            return GraspRetryStep(action=action, phase=self.phase)
        if self.phase == "released":
            return GraspRetryStep(action=action, phase=self.phase)
        if self.phase == "disabled":
            target = tcp if self._abort_xyz is None else self._abort_xyz
            result = self._override(
                action, tcp, target, self.config.open_gripper_value
            )
            return GraspRetryStep(
                action=result.action,
                abort_attempt=True,
                phase=self.phase,
            )

        if self.phase == "tracking":
            if float(action[-1]) > self.config.close_command_threshold:
                return GraspRetryStep(action=action, phase=self.phase)
            self.phase = "observing_close"
            self.phase_step = 1
            return GraspRetryStep(
                action=action,
                event="grasp_check_started",
                phase=self.phase,
            )

        if self.phase == "observing_close":
            if float(action[-1]) > self.config.close_command_threshold:
                self.phase = "tracking"
                self.phase_step = 0
                return GraspRetryStep(
                    action=action,
                    event="grasp_check_cancelled",
                    phase=self.phase,
                )
            if self.phase_step < self.config.verification_steps:
                self.phase_step += 1
                return GraspRetryStep(action=action, phase=self.phase)
            if qpos >= self.config.grasp_qpos_threshold:
                self.phase = "verified"
                self.grasp_verified = True
                return GraspRetryStep(
                    action,
                    event="grasp_verified",
                    phase=self.phase,
                )
            return self._begin_recovery(
                action, tcp, reason="empty_close"
            )

        if self.phase == "opening":
            assert self._recovery_start_xyz is not None
            if self.phase_step >= self.config.open_steps:
                self.phase = "backing_off"
                self.phase_step = 0
                event = "gripper_opened"
            else:
                self.phase_step += 1
                return self._override(
                    action,
                    tcp,
                    self._recovery_start_xyz,
                    self.config.open_gripper_value,
                )
            self.phase_step = 1
            target = self._recovery_start_xyz.copy()
            if self.config.backoff_steps > 0:
                target[2] += self.config.backoff_height / self.config.backoff_steps
            return self._override(
                action,
                tcp,
                target,
                self.config.open_gripper_value,
                event=event,
            )

        if self.phase == "backing_off":
            assert self._recovery_start_xyz is not None
            if self.phase_step >= self.config.backoff_steps:
                use_visual = (
                    not self._disable_after_recovery
                    and self.config.recovery_mode == "rgb"
                    and self._valid_visual_target(visual_target_xyz)
                )
                if use_visual:
                    self._visual_cube_xyz = np.asarray(
                        visual_target_xyz, dtype=np.float64
                    ).copy()
                    self.phase = "visual_approach"
                    self._settled_steps = 0
                    event = "rgb_retry_reacquired"
                else:
                    self.phase = (
                        "disabled" if self._disable_after_recovery else "tracking"
                    )
                    event = (
                        "local_retries_exhausted"
                        if self.phase == "disabled"
                        else "retry_replan"
                    )
                self.phase_step = 0
                if self.phase == "disabled":
                    self._abort_xyz = tcp.copy()
                return GraspRetryStep(
                    action=self._target_action(
                        action,
                        tcp,
                        tcp,
                        self.config.open_gripper_value,
                    ),
                    reset_policy=True,
                    abort_attempt=self.phase == "disabled",
                    event=event,
                    phase=self.phase,
                )
            self.phase_step += 1
            frac = self.phase_step / max(1, self.config.backoff_steps)
            target = self._recovery_start_xyz.copy()
            target[2] += self.config.backoff_height * min(frac, 1.0)
            return self._override(
                action, tcp, target, self.config.open_gripper_value
            )

        if self.phase == "visual_approach":
            target = self._visual_target(approach=True)
            self.phase_step += 1
            settled = self._target_is_settled(tcp, target)
            if settled or self.phase_step >= self.config.visual_approach_steps:
                self.phase = "visual_descent"
                self.phase_step = 0
                self._settled_steps = 0
                event = "rgb_retry_descend"
            else:
                event = None
            return self._override(
                action,
                tcp,
                target,
                self.config.open_gripper_value,
                event=event,
            )

        if self.phase == "visual_descent":
            target = self._visual_target(approach=False)
            self.phase_step += 1
            settled = self._target_is_settled(tcp, target)
            if settled or self.phase_step >= self.config.visual_descent_steps:
                self.phase = "visual_close"
                self.phase_step = 0
                self._settled_steps = 0
                # Capture on the next frame, after this final descent target
                # has actually been applied by the robot.
                self._visual_close_anchor_xyz = None
                self._obstruction_steps = 0
                event = "rgb_retry_close"
            else:
                event = None
            return self._override(
                action,
                tcp,
                target,
                self.config.open_gripper_value,
                event=event,
            )

        if self.phase == "visual_close":
            if self._visual_close_anchor_xyz is None:
                self._visual_close_anchor_xyz = tcp.copy()
            target = self._visual_close_anchor_xyz
            self.phase_step += 1
            ramp_steps = max(1, self.config.visual_close_steps)
            frac = min(self.phase_step / ramp_steps, 1.0)
            gripper = (
                self.config.open_gripper_value
                + frac * (
                    self.config.probe_gripper_value
                    - self.config.open_gripper_value
                )
            )
            if (
                gripper <= self.config.close_command_threshold
                and qpos >= self.config.grasp_qpos_threshold
            ):
                self._obstruction_steps += 1
            else:
                self._obstruction_steps = 0
            if self._obstruction_steps >= self.config.verification_steps:
                self.phase = "visual_tighten"
                self.phase_step = 0
                self._settled_steps = 0
                self.grasp_verified = True
                self._lost_grasp_steps = 0
                self._grasp_contact_value = gripper
                self._grasp_hold_value = max(
                    self.config.closed_gripper_value,
                    self._grasp_contact_value
                    - self.config.visual_tighten_delta,
                )
                return self._override(
                    action,
                    tcp,
                    target,
                    self._grasp_contact_value,
                    event="rgb_grasp_detected",
                )
            decision_step = ramp_steps + self.config.verification_steps
            if self.phase_step < decision_step:
                return self._override(action, tcp, target, gripper)
            return self._begin_recovery(
                action, tcp, reason="rgb_empty_close"
            )

        if self.phase == "visual_tighten":
            assert self._visual_close_anchor_xyz is not None
            target = self._visual_close_anchor_xyz
            self.phase_step += 1
            frac = min(
                self.phase_step / max(1, self.config.visual_tighten_steps),
                1.0,
            )
            gripper = (
                self._grasp_contact_value
                + frac * (
                    self._grasp_hold_value
                    - self._grasp_contact_value
                )
            )
            if qpos < self.config.grasp_qpos_threshold:
                self._lost_grasp_steps += 1
                if self._lost_grasp_steps >= self.config.verification_steps:
                    return self._begin_recovery(
                        action, tcp, reason="grasp_lost_during_tighten"
                    )
            else:
                self._lost_grasp_steps = 0
            if self.phase_step >= self.config.visual_tighten_steps:
                self.phase = "visual_lift"
                self.phase_step = 0
                self._settled_steps = 0
                event = "rgb_grasp_secured"
            else:
                event = None
            return self._override(
                action,
                tcp,
                target,
                gripper,
                event=event,
            )

        if self.phase == "visual_lift":
            assert self._visual_cube_xyz is not None
            target = self._visual_target(approach=False)
            target[2] = (
                self._visual_cube_xyz[2]
                + self.config.visual_lift_z_offset
            )
            if qpos < self.config.grasp_qpos_threshold:
                self._lost_grasp_steps += 1
                if self._lost_grasp_steps >= self.config.verification_steps:
                    return self._begin_recovery(
                        action, tcp, reason="grasp_lost_during_lift"
                    )
            else:
                self._lost_grasp_steps = 0
            self.phase_step += 1
            settled = self._target_is_settled(tcp, target)
            if settled or self.phase_step >= self.config.visual_lift_steps:
                self.phase = "verified"
                self.phase_step = 0
                return self._override(
                    action,
                    tcp,
                    target,
                    self._grasp_hold_value,
                    reset_policy=True,
                    event="grasp_verified",
                )
            return self._override(
                action,
                tcp,
                target,
                self._grasp_hold_value,
            )

        raise RuntimeError(f"unknown local grasp retry phase {self.phase!r}")
