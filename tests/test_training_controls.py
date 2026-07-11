import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


class TrainingControlsTest(unittest.TestCase):
    def test_bc_loss_weights_emphasize_gripper_and_late_phase(self):
        from scripts.train import build_bc_loss_weights

        phase = torch.tensor([0.10, 0.45, 0.80])
        mask = torch.ones(3, 4)

        sample_w, dim_w, denom = build_bc_loss_weights(
            phase=phase,
            mask=mask,
            action_dim=6,
            early_bc_frac=0.22,
            early_bc_weight=2.0,
            late_bc_frac=0.35,
            late_bc_weight=5.0,
            gripper_loss_weight=7.0,
        )

        self.assertEqual(sample_w[:, 0, 0].tolist(), [2.0, 5.0, 5.0])
        self.assertEqual(dim_w.tolist(), [1.0, 1.0, 1.0, 1.0, 1.0, 7.0])
        self.assertAlmostEqual(denom.item(), (2.0 + 5.0 + 5.0) * 4 * 12.0)


    def test_recovery_offset_decays_before_final_grasp(self):
        from bude_vla.scripted_pick_and_place import decaying_recovery_offset

        offset = torch.tensor([0.02, -0.01]).numpy()

        self.assertEqual(decaying_recovery_offset(offset, 0, 100).round(6).tolist(), [0.02, -0.01])
        self.assertEqual(decaying_recovery_offset(offset, 50, 100).round(6).tolist(), [0.01, -0.005])
        self.assertEqual(decaying_recovery_offset(offset, 100, 100).round(6).tolist(), [0.0, -0.0])
        self.assertEqual(decaying_recovery_offset(offset, 150, 100).round(6).tolist(), [0.0, -0.0])


    def test_recovery_scalar_decays_before_final_grasp(self):
        from bude_vla.scripted_pick_and_place import decaying_recovery_scalar

        self.assertEqual(round(decaying_recovery_scalar(0.012, 0, 100), 6), 0.012)
        self.assertEqual(round(decaying_recovery_scalar(0.012, 50, 100), 6), 0.006)
        self.assertEqual(round(decaying_recovery_scalar(0.012, 100, 100), 6), 0.0)
        self.assertEqual(round(decaying_recovery_scalar(0.012, 150, 100), 6), 0.0)


    def test_phase_names_include_nudge_backoff_recovery(self):
        from bude_vla.scripted_pick_and_place import BACKOFF, PHASE_NAMES

        self.assertEqual(PHASE_NAMES[BACKOFF], "BACKOFF")


    def test_failed_close_retries_until_retry_budget_is_spent(self):
        from bude_vla.scripted_pick_and_place import should_retry_close

        self.assertTrue(should_retry_close(contact_step=None, retries_used=0, max_retries=1))
        self.assertFalse(should_retry_close(contact_step=12, retries_used=0, max_retries=1))
        self.assertFalse(should_retry_close(contact_step=None, retries_used=1, max_retries=1))


    def test_resolve_frame_caches_matches_multiple_roots(self):
        from scripts.train import resolve_frame_caches

        roots = ["data/a", "data/b"]

        self.assertEqual(resolve_frame_caches(roots, None), [None, None])
        self.assertEqual(resolve_frame_caches(roots, "cache/a:cache/b"), ["cache/a", "cache/b"])
        with self.assertRaisesRegex(ValueError, "same number"):
            resolve_frame_caches(roots, "cache/a")


    def test_merge_action_stats_uses_per_dimension_union(self):
        import numpy as np
        from scripts.train import merge_action_stats

        lo, hi = merge_action_stats([
            (np.array([-1.0, -0.5]), np.array([0.5, 2.0])),
            (np.array([-0.2, -1.5]), np.array([1.0, 1.2])),
        ])

        self.assertEqual(lo.tolist(), [-1.0, -1.5])
        self.assertEqual(hi.tolist(), [1.0, 2.0])



    def test_action_dim_weights_emphasize_gripper_for_flow_and_bc(self):
        from scripts.train import build_action_dim_weights

        w = build_action_dim_weights(6, torch.device("cpu"), torch.float32, 5.0)

        self.assertEqual(w.tolist(), [1.0, 1.0, 1.0, 1.0, 1.0, 5.0])

    def test_parse_cube_positions_accepts_explicit_reachable_eval_set(self):
        from scripts.eval_pick_ball import parse_cube_positions

        positions = parse_cube_positions("0.25,0.00;0.30,-0.04;0.30,0.06")

        self.assertEqual(positions, [(0.25, 0.0), (0.30, -0.04), (0.30, 0.06)])

    def test_parse_cube_positions_rejects_bad_pairs(self):
        from scripts.eval_pick_ball import parse_cube_positions

        with self.assertRaisesRegex(ValueError, "x,y"):
            parse_cube_positions("0.25;0.30,0.04")

    def test_parse_weighted_phase_ranges_for_contact_focused_cache(self):
        from scripts.build_frame_cache import parse_weighted_phase_ranges

        ranges = parse_weighted_phase_ranges("0.08:0.28:0.70,0.28:0.55:0.25,0.55:1.00:0.05")

        self.assertEqual(ranges, [(0.08, 0.28, 0.70), (0.28, 0.55, 0.25), (0.55, 1.00, 0.05)])
        with self.assertRaisesRegex(ValueError, "lo:hi:weight"):
            parse_weighted_phase_ranges("0.1:0.2")
        with self.assertRaisesRegex(ValueError, "0 <= lo < hi <= 1"):
            parse_weighted_phase_ranges("0.4:0.2:1")

    def test_select_contact_focused_frames_oversamples_contact_indices(self):
        import numpy as np
        from scripts.build_frame_cache import EpisodeInfo, select_cache_frame_indices

        episodes = [EpisodeInfo(ep_idx=0, chunk_idx=0, start=0, length=10, contact_locals=[5])]
        selected = select_cache_frame_indices(
            episodes,
            max_frames=6,
            rng=np.random.default_rng(0),
            early_prob=0.0,
            early_max_frac=0.22,
            phase_bins=0,
            phase_ranges=[],
            contact_prob=1.0,
            contact_jitter=0,
        )

        self.assertEqual(selected[0], {5})

    def test_select_contact_focused_frames_falls_back_without_contacts(self):
        import numpy as np
        from scripts.build_frame_cache import EpisodeInfo, select_cache_frame_indices

        episodes = [EpisodeInfo(ep_idx=0, chunk_idx=0, start=0, length=10, contact_locals=[])]
        selected = select_cache_frame_indices(
            episodes,
            max_frames=3,
            rng=np.random.default_rng(1),
            early_prob=0.0,
            early_max_frac=0.22,
            phase_bins=0,
            phase_ranges=[],
            contact_prob=1.0,
            contact_jitter=0,
        )

        self.assertGreater(len(selected[0]), 0)


    def test_dinov2_adapter_initializes_current_top_frame(self):
        from bude_vla.models.vision import current_top_rgb_start

        self.assertEqual(current_top_rgb_start(3), 0)
        self.assertEqual(current_top_rgb_start(6), 0)
        self.assertEqual(current_top_rgb_start(12), 6)
        self.assertEqual(current_top_rgb_start(24), 18)
        with self.assertRaises(ValueError):
            current_top_rgb_start(9)

    def test_red_detector_does_not_average_separate_objects(self):
        import numpy as np
        from bude_vla.perception import detect_red_candidates, detect_red_centroid

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[8:14, 8:14, 0] = 240
        image[38:50, 42:54, 0] = 240

        candidates = detect_red_candidates(image)
        centroid = detect_red_centroid(image)

        self.assertEqual(candidates.shape[0], 2)
        self.assertAlmostEqual(float(centroid[0]), 2.0 * 47.5 / 63.0 - 1.0, places=2)
        self.assertAlmostEqual(float(centroid[1]), 2.0 * 43.5 / 63.0 - 1.0, places=2)
        self.assertEqual(float(centroid[2]), 1.0)

    def test_planar_homography_recovers_projective_mapping(self):
        import numpy as np
        from bude_vla.visual_servo import PlanarHomography

        true_h = np.asarray([
            [0.22, 0.01, 0.27],
            [-0.02, -0.18, 0.03],
            [0.04, -0.03, 1.0],
        ])
        image = np.asarray([
            [-0.8, -0.7], [-0.8, 0.7], [0.0, -0.7],
            [0.0, 0.7], [0.8, -0.7], [0.8, 0.7],
        ])
        homogeneous = np.concatenate([image, np.ones((len(image), 1))], axis=1)
        projected = homogeneous @ true_h.T
        world = projected[:, :2] / projected[:, 2:3]

        fitted = PlanarHomography.fit(image, world)

        self.assertLess(float(fitted.errors(image, world).max()), 1e-10)

    def test_grasp_contact_requires_recent_contact(self):
        from bude_vla.scripted_pick_and_place import has_recent_grasp_contact

        self.assertTrue(has_recent_grasp_contact(20, 12, grace_steps=8))
        self.assertFalse(has_recent_grasp_contact(21, 12, grace_steps=8))
        self.assertFalse(has_recent_grasp_contact(20, None, grace_steps=8))

    def test_policy_rate_matches_record_decimation(self):
        from bude_vla.envs.so101_mjx import (
            EXPERT_CONTROL_SUBSTEPS,
            POLICY_CONTROL_SUBSTEPS,
            POLICY_RECORD_STRIDE,
        )

        self.assertEqual(
            POLICY_CONTROL_SUBSTEPS,
            EXPERT_CONTROL_SUBSTEPS * POLICY_RECORD_STRIDE,
        )

    def test_policy_clipping_uses_loaded_robot_limits(self):
        import numpy as np
        from bude_vla.action_space import (
            clip_arm_joint_targets,
            clip_gripper_control,
        )
        from bude_vla.envs.so101_mjx import load_arm_model

        model = load_arm_model()
        arm = clip_arm_joint_targets(model, np.full(5, 100.0))
        for idx, value in enumerate(arm):
            lo, hi = model.jnt_range[idx]
            if hi > lo:
                self.assertLessEqual(float(value), float(hi))
                self.assertGreaterEqual(float(value), float(lo))

        grip_lo, grip_hi = model.actuator_ctrlrange[5]
        self.assertEqual(clip_gripper_control(model, 100.0), float(grip_hi))
        self.assertEqual(clip_gripper_control(model, -100.0), float(grip_lo))

if __name__ == "__main__":
    unittest.main()
