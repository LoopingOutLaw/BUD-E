import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


class TrainingControlsTest(unittest.TestCase):
    def test_affine_feature_transform_preserves_layernorm_collision(self):
        from bude_vla.models.proprio import PerFeatureAffine

        # x2 = a*x1 + b for every feature, so LayerNorm maps these distinct
        # cube centroids to the same vector. The affine transform must not.
        x1 = torch.tensor([[0.20, 0.10, 1.00]])
        x2 = torch.tensor([[0.04, -0.08, 1.00]])

        self.assertTrue(torch.allclose(
            torch.nn.LayerNorm(3, elementwise_affine=False)(x1),
            torch.nn.LayerNorm(3, elementwise_affine=False)(x2),
            atol=1e-5,
        ))
        affine = PerFeatureAffine(3)
        self.assertFalse(torch.allclose(affine(x1), affine(x2)))

    def test_affine_policy_keeps_checkpoint_compatible_parameter_shapes(self):
        from bude_vla.models.policy import BUDEConfig, BUDEPolicy

        legacy_cfg = BUDEConfig(use_perception=True, input_feature_norm="layernorm")
        fixed_cfg = BUDEConfig(use_perception=True, input_feature_norm="affine")
        legacy = BUDEPolicy(legacy_cfg)
        fixed = BUDEPolicy(fixed_cfg)

        self.assertEqual(
            {k: tuple(v.shape) for k, v in legacy.state_dict().items()},
            {k: tuple(v.shape) for k, v in fixed.state_dict().items()},
        )

    def test_saved_config_applies_new_architecture_fields(self):
        from bude_vla.models.policy import BUDEConfig, apply_saved_config

        cfg = apply_saved_config(BUDEConfig(), {
            "chunk_size": 16,
            "action_space": "ee_abs",
            "use_raw_geometry_action_cond": True,
            "unrelated_training_field": 123,
        })

        self.assertEqual(cfg.chunk_size, 16)
        self.assertEqual(cfg.action_space, "ee_abs")
        self.assertTrue(cfg.use_raw_geometry_action_cond)
        self.assertFalse(hasattr(cfg, "unrelated_training_field"))

    def test_raw_geometry_residual_is_zero_initialized(self):
        from bude_vla.models.action_head import ContextActionHead

        torch.manual_seed(3)
        head = ContextActionHead(
            action_dim=4, chunk_size=3, d=16, hidden_dim=32,
            depth=1, heads=4, cond_dim=0, raw_cond_dim=9,
        )
        tokens = torch.randn(2, 5, 16)
        raw_a = torch.zeros(2, 9)
        raw_b = torch.ones(2, 9)

        out_a = head(tokens, raw_cond=raw_a)
        out_b = head(tokens, raw_cond=raw_b)

        self.assertTrue(torch.allclose(out_a, out_b))
        self.assertTrue(torch.count_nonzero(
            head.raw_geometry_residual[-1].weight
        ).item() == 0)

    def test_raw_geometry_requires_context_decoder(self):
        from bude_vla.models.policy import BUDEConfig, BUDEPolicy

        cfg = BUDEConfig(
            use_perception=True,
            use_context_action_head=False,
            use_raw_geometry_action_cond=True,
        )
        with self.assertRaisesRegex(ValueError, "context action head"):
            BUDEPolicy(cfg)

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
            shoulder_pan_loss_weight=3.0,
            shoulder_lift_loss_weight=7.0,
        )

        self.assertEqual(sample_w[:, 0, 0].tolist(), [2.0, 5.0, 5.0])
        self.assertEqual(dim_w.tolist(), [3.0, 7.0, 1.0, 1.0, 1.0, 7.0])
        self.assertAlmostEqual(denom.item(), (2.0 + 5.0 + 5.0) * 4 * 20.0)

    def test_bc_loss_weights_emphasize_chunk_endpoint(self):
        from scripts.train import build_bc_loss_weights

        sample_w, _dim_w, denom = build_bc_loss_weights(
            phase=None,
            mask=torch.ones(1, 4),
            action_dim=2,
            early_bc_frac=0.2,
            early_bc_weight=1.0,
            late_bc_frac=0.5,
            late_bc_weight=1.0,
            gripper_loss_weight=1.0,
            chunk_end_bc_weight=4.0,
        )

        self.assertEqual(sample_w[0, :, 0].tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(denom.item(), 20.0)

    def test_bc_error_supports_l1(self):
        from scripts.train import build_bc_error

        prediction = torch.tensor([0.0, 2.0])
        target = torch.tensor([1.0, -1.0])
        self.assertEqual(
            build_bc_error(prediction, target, "l1").tolist(), [1.0, 3.0]
        )


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

    def test_close_anchor_does_not_chase_a_displaced_cube(self):
        import mujoco
        import numpy as np
        from bude_vla.envs.so101_mjx import CUBE_QPOS_START, CUBE_REST_Z, load_arm_model
        from bude_vla.scripted_pick_and_place import ScriptedPickAndPlace

        model = load_arm_model()
        data = mujoco.MjData(model)
        data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = [0.27, 0.02, CUBE_REST_Z]
        mujoco.mj_forward(model, data)
        expert = ScriptedPickAndPlace(model, data, np.array([0.27, 0.02]))
        expert._begin_close(data)
        anchor = expert._close_anchor_xyz.copy()

        data.qpos[CUBE_QPOS_START:CUBE_QPOS_START + 3] = [0.24, -0.05, CUBE_REST_Z]
        mujoco.mj_forward(model, data)

        np.testing.assert_allclose(expert._close_anchor_xyz, anchor)
        self.assertGreater(float(np.linalg.norm(expert._cube_xyz(data) - anchor)), 0.05)


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



    def test_action_dim_weights_emphasize_shoulder_and_gripper(self):
        from scripts.train import build_action_dim_weights

        w = build_action_dim_weights(
            6, torch.device("cpu"), torch.float32, 5.0, 9.0, 8.0
        )

        self.assertEqual(w.tolist(), [9.0, 8.0, 1.0, 1.0, 1.0, 5.0])

    def test_parse_cube_positions_accepts_explicit_reachable_eval_set(self):
        from scripts.eval_pick_ball import parse_cube_positions

        positions = parse_cube_positions("0.25,0.00;0.30,-0.04;0.30,0.06")

        self.assertEqual(positions, [(0.25, 0.0), (0.30, -0.04), (0.30, 0.06)])

    def test_parse_cube_positions_rejects_bad_pairs(self):
        from scripts.eval_pick_ball import parse_cube_positions

        with self.assertRaisesRegex(ValueError, "x,y"):
            parse_cube_positions("0.25;0.30,0.04")

    def test_parse_weighted_phase_ranges_for_contact_focused_cache(self):
        from scripts.build_frame_cache import (
            parse_anchor_local_frames,
            parse_weighted_phase_ranges,
        )

        ranges = parse_weighted_phase_ranges("0.08:0.28:0.70,0.28:0.55:0.25,0.55:1.00:0.05")

        self.assertEqual(ranges, [(0.08, 0.28, 0.70), (0.28, 0.55, 0.25), (0.55, 1.00, 0.05)])
        with self.assertRaisesRegex(ValueError, "lo:hi:weight"):
            parse_weighted_phase_ranges("0.1:0.2")
        with self.assertRaisesRegex(ValueError, "0 <= lo < hi <= 1"):
            parse_weighted_phase_ranges("0.4:0.2:1")
        self.assertEqual(parse_anchor_local_frames("0,1,4,4,16"), [0, 1, 4, 16])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            parse_anchor_local_frames("0,-1")

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


    def test_cache_min_frames_covers_every_episode(self):
        import numpy as np
        from scripts.build_frame_cache import EpisodeInfo, select_cache_frame_indices

        episodes = [
            EpisodeInfo(ep_idx=i, chunk_idx=0, start=i * 100, length=100, contact_locals=[])
            for i in range(3)
        ]
        selected = select_cache_frame_indices(
            episodes,
            max_frames=12,
            rng=np.random.default_rng(7),
            early_prob=0.0,
            early_max_frac=0.22,
            phase_bins=0,
            phase_ranges=[],
            contact_prob=0.0,
            contact_jitter=0,
            min_frames_per_episode=3,
        )

        self.assertTrue(all(len(selected[i]) >= 3 for i in range(3)))
        self.assertLessEqual(sum(map(len, selected.values())), 12)

    def test_cache_anchors_exact_reset_frames_for_every_episode(self):
        import numpy as np
        from scripts.build_frame_cache import EpisodeInfo, select_cache_frame_indices

        episodes = [
            EpisodeInfo(ep_idx=i, chunk_idx=0, start=i * 50, length=50, contact_locals=[])
            for i in range(3)
        ]
        selected = select_cache_frame_indices(
            episodes,
            max_frames=12,
            rng=np.random.default_rng(9),
            early_prob=0.0,
            early_max_frac=0.2,
            phase_bins=0,
            phase_ranges=[],
            contact_prob=0.0,
            contact_jitter=0,
            anchor_local_frames=[0, 1, 4],
        )

        self.assertTrue(all({0, 1, 4}.issubset(selected[i]) for i in range(3)))

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

    def test_joint_action_to_ee_abs_preserves_tcp_target_and_gripper(self):
        import numpy as np
        import mujoco
        from bude_vla.action_space import (
            end_effector_position_for_qpos,
            joint_action_to_ee_abs,
        )
        from bude_vla.envs.so101_mjx import load_arm_model

        model = load_arm_model()
        fk_data = mujoco.MjData(model)
        action = np.asarray(
            [0.12, -0.63, 0.71, np.pi / 2, np.pi / 2, -0.4],
            dtype=np.float64,
        )

        converted = joint_action_to_ee_abs(model, fk_data, action)
        expected_xyz = end_effector_position_for_qpos(
            model, fk_data, action[:5], action[5]
        )

        np.testing.assert_allclose(converted[:3], expected_xyz, atol=1e-6)
        self.assertAlmostEqual(float(converted[3]), float(action[5]), places=6)


    def test_ee_abs_is_an_ik_action_space(self):
        from types import SimpleNamespace
        from bude_vla.action_space import uses_ik_action_space

        self.assertTrue(uses_ik_action_space(SimpleNamespace(action_space="ee_abs")))
        self.assertTrue(uses_ik_action_space(SimpleNamespace(action_space="ee_delta")))
        self.assertFalse(uses_ik_action_space(SimpleNamespace(action_space="joint_abs")))


if __name__ == "__main__":
    unittest.main()
