import unittest

import numpy as np

from bude_vla.grasp_retry import (
    LocalGraspRetryConfig,
    LocalGraspRetryController,
)


def _action(gripper: float, xyz=(0.25, 0.0, 0.04)) -> np.ndarray:
    return np.asarray([*xyz, gripper], dtype=np.float32)


def _advance_to_verification(
    controller: LocalGraspRetryController,
    qpos: float,
):
    tcp = np.asarray([0.25, 0.0, 0.04])
    result = controller.step(_action(-0.2), tcp, 0.3)
    if result.event != "grasp_check_started":
        raise AssertionError(result.event)
    for _ in range(controller.config.verification_steps - 1):
        result = controller.step(_action(-0.2), tcp, qpos)
    return controller.step(_action(-0.2), tcp, qpos)


class LocalGraspRetryTest(unittest.TestCase):
    def test_verified_grasp_leaves_every_policy_action_untouched(self):
        cfg = LocalGraspRetryConfig(verification_steps=2)
        controller = LocalGraspRetryController("ee_abs", cfg)
        tcp = np.asarray([0.25, 0.0, 0.04])
        close = _action(-0.2, xyz=(0.26, 0.01, 0.03))

        first = controller.step(close, tcp, 0.3)
        second = controller.step(close, tcp, 0.16)
        third = controller.step(close, tcp, 0.16)
        np.testing.assert_array_equal(first.action, close)
        np.testing.assert_array_equal(second.action, close)
        np.testing.assert_array_equal(third.action, close)
        self.assertEqual(third.event, "grasp_verified")
        self.assertEqual(controller.phase, "verified")
        self.assertFalse(controller.blocks_policy)

        release = _action(0.8, xyz=(0.32, 0.16, 0.08))
        result = controller.step(release, tcp, 0.15)
        np.testing.assert_array_equal(result.action, release)

    def test_empty_close_opens_backs_off_and_replans_locally(self):
        cfg = LocalGraspRetryConfig(
            max_retries=1,
            verification_steps=2,
            open_steps=2,
            backoff_steps=2,
            backoff_height=0.02,
        )
        controller = LocalGraspRetryController("ee_abs", cfg)
        tcp = np.asarray([0.25, 0.0, 0.04])

        result = _advance_to_verification(controller, qpos=-0.17)
        self.assertEqual(result.event, "empty_close_retry_1")
        self.assertEqual(result.phase, "opening")
        self.assertTrue(result.reset_policy)
        self.assertTrue(controller.blocks_policy)
        self.assertAlmostEqual(float(result.action[-1]), cfg.open_gripper_value)

        events = []
        targets = []
        for _ in range(8):
            result = controller.step(_action(-0.2), tcp, -0.17)
            targets.append(result.action[:3].copy())
            if result.event:
                events.append(result.event)
            if result.event == "retry_replan":
                break

        self.assertIn("gripper_opened", events)
        self.assertEqual(events[-1], "retry_replan")
        self.assertEqual(controller.phase, "tracking")
        self.assertEqual(controller.retries_used, 1)
        self.assertFalse(controller.blocks_policy)
        self.assertGreaterEqual(
            max(target[2] for target in targets),
            tcp[2] + cfg.backoff_height - 1e-7,
        )

    def test_exhausted_retry_aborts_instead_of_transporting_empty(self):
        cfg = LocalGraspRetryConfig(
            max_retries=0,
            verification_steps=1,
            open_steps=0,
            backoff_steps=0,
        )
        controller = LocalGraspRetryController("ee_abs", cfg)
        tcp = np.asarray([0.25, 0.0, 0.04])
        controller.step(_action(-0.2), tcp, 0.3)
        controller.step(_action(-0.2), tcp, -0.17)
        controller.step(_action(-0.2), tcp, -0.17)
        result = controller.step(_action(-0.2), tcp, -0.17)

        self.assertEqual(result.event, "local_retries_exhausted")
        self.assertTrue(result.abort_attempt)
        self.assertEqual(controller.phase, "disabled")
        self.assertAlmostEqual(float(result.action[-1]), cfg.open_gripper_value)

    def test_ee_delta_recovery_override_is_bounded(self):
        cfg = LocalGraspRetryConfig(
            verification_steps=1,
            ee_delta_scale=0.01,
        )
        controller = LocalGraspRetryController("ee_delta", cfg)
        tcp = np.asarray([0.25, 0.0, 0.04])
        action = _action(-0.2, xyz=(1.0, 1.0, 1.0))
        controller.step(action, tcp, 0.3)
        result = controller.step(action, tcp, -0.17)
        self.assertEqual(result.event, "empty_close_retry_1")
        np.testing.assert_allclose(result.action[:3], np.zeros(3), atol=1e-7)

    def test_transient_close_request_is_cancelled(self):
        cfg = LocalGraspRetryConfig(verification_steps=2)
        controller = LocalGraspRetryController("ee_abs", cfg)
        tcp = np.asarray([0.25, 0.0, 0.04])

        controller.step(_action(-0.2), tcp, 0.3)
        result = controller.step(_action(0.4), tcp, 0.3)

        self.assertEqual(result.event, "grasp_check_cancelled")
        self.assertEqual(controller.phase, "tracking")
        np.testing.assert_array_equal(result.action, _action(0.4))

    def test_verified_grasp_loss_starts_local_recovery(self):
        cfg = LocalGraspRetryConfig(
            max_retries=1,
            verification_steps=1,
        )
        controller = LocalGraspRetryController("ee_abs", cfg)
        tcp = np.asarray([0.25, 0.0, 0.04])
        verified = _advance_to_verification(controller, qpos=0.16)
        self.assertEqual(verified.event, "grasp_verified")

        result = controller.step(_action(-0.2), tcp, -0.17)

        self.assertEqual(result.event, "grasp_lost_retry_1")
        self.assertEqual(controller.phase, "opening")
        self.assertTrue(result.reset_policy)

    def test_rgb_retry_reacquires_closes_lifts_then_returns_to_policy(self):
        cfg = LocalGraspRetryConfig(
            max_retries=1,
            verification_steps=1,
            open_steps=0,
            backoff_steps=0,
            visual_approach_steps=1,
            visual_descent_steps=1,
            visual_close_steps=2,
            visual_tighten_steps=1,
            visual_lift_steps=1,
            visual_settle_steps=1,
            visual_target_tolerance=1.0,
        )
        controller = LocalGraspRetryController("ee_abs", cfg)
        tcp = np.asarray([0.25, 0.0, 0.04])
        cube = np.asarray([0.30, 0.02, 0.015])
        close = _action(-0.2)

        failed = _advance_to_verification(controller, qpos=-0.17)
        self.assertEqual(failed.event, "empty_close_retry_1")
        controller.step(close, tcp, -0.17, visual_target_xyz=cube)
        reacquired = controller.step(
            close, tcp, -0.17, visual_target_xyz=cube
        )
        self.assertEqual(reacquired.event, "rgb_retry_reacquired")

        descent = controller.step(close, tcp, -0.17, visual_target_xyz=cube)
        self.assertEqual(descent.event, "rgb_retry_descend")
        close_start = controller.step(
            close, tcp, -0.17, visual_target_xyz=cube
        )
        self.assertEqual(close_start.event, "rgb_retry_close")
        controller.step(close, tcp, -0.17, visual_target_xyz=cube)
        detected = controller.step(close, tcp, 0.16, visual_target_xyz=cube)
        self.assertEqual(detected.event, "rgb_grasp_detected")
        secured = controller.step(close, tcp, 0.16, visual_target_xyz=cube)
        self.assertEqual(secured.event, "rgb_grasp_secured")
        verified = controller.step(close, tcp, 0.16, visual_target_xyz=cube)
        self.assertEqual(verified.event, "grasp_verified")
        self.assertEqual(controller.phase, "verified")
        self.assertTrue(verified.reset_policy)
        self.assertFalse(controller.blocks_policy)


if __name__ == "__main__":
    unittest.main()
