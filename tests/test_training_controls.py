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


    def test_failed_close_retries_until_retry_budget_is_spent(self):
        from bude_vla.scripted_pick_and_place import should_retry_close

        self.assertTrue(should_retry_close(contact_step=None, retries_used=0, max_retries=1))
        self.assertFalse(should_retry_close(contact_step=12, retries_used=0, max_retries=1))
        self.assertFalse(should_retry_close(contact_step=None, retries_used=1, max_retries=1))

    def test_parse_cube_positions_accepts_explicit_reachable_eval_set(self):
        from scripts.eval_pick_ball import parse_cube_positions

        positions = parse_cube_positions("0.25,0.00;0.30,-0.04;0.30,0.06")

        self.assertEqual(positions, [(0.25, 0.0), (0.30, -0.04), (0.30, 0.06)])

    def test_parse_cube_positions_rejects_bad_pairs(self):
        from scripts.eval_pick_ball import parse_cube_positions

        with self.assertRaisesRegex(ValueError, "x,y"):
            parse_cube_positions("0.25;0.30,0.04")


if __name__ == "__main__":
    unittest.main()
