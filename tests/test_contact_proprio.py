import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ContactProprioTest(unittest.TestCase):
    def _fake_model_ids(self, _model, _obj_type, name):
        ids = {
            "cube_geom": 10,
            "static_finger_pad": 20,
            "moving_finger_pad": 30,
            "gripperframe": 40,
            "target_zone": 50,
        }
        return ids[name]

    def test_any_pad_contact_is_true_before_strict_grasp(self):
        from bude_vla.envs.so101_mjx import (
            is_grasping_from_contacts,
            is_touching_cube_from_contacts,
        )

        data = SimpleNamespace(
            ncon=1,
            contact=[SimpleNamespace(geom1=10, geom2=20)],
        )

        with patch("bude_vla.envs.so101_mjx.mujoco.mj_name2id", self._fake_model_ids):
            self.assertEqual(is_touching_cube_from_contacts(object(), data), 1.0)
            self.assertEqual(is_grasping_from_contacts(object(), data), 0.0)

    def test_build_pick_proprio_10d_adds_any_contact_before_grasp(self):
        from bude_vla.envs.so101_mjx import build_pick_proprio

        data = SimpleNamespace(
            ncon=1,
            contact=[SimpleNamespace(geom1=10, geom2=20)],
            qpos=np.array([1, 2, 3, 4, 5, 6, 0, 0, 0, 0], dtype=np.float64),
            site_xpos=np.zeros((64, 3), dtype=np.float64),
            xpos=np.zeros((64, 3), dtype=np.float64),
        )
        data.site_xpos[40, :2] = [0.2, 0.1]
        data.xpos[50, :2] = [0.5, 0.0]

        with patch("bude_vla.envs.so101_mjx.mujoco.mj_name2id", self._fake_model_ids):
            proprio = build_pick_proprio(object(), data, state_dim=10)

        self.assertEqual(proprio.shape, (10,))
        self.assertEqual(proprio[:6].tolist(), [1, 2, 3, 4, 5, 6])
        self.assertAlmostEqual(float(proprio[6]), 0.3, places=6)
        self.assertAlmostEqual(float(proprio[7]), -0.1, places=6)
        self.assertEqual(float(proprio[8]), 1.0)  # any pad contact
        self.assertEqual(float(proprio[9]), 0.0)  # strict two-pad grasp

    def test_adapt_state_dict_splits_legacy_grasp_column_for_10d_proprio(self):
        from scripts.train import adapt_state_dict_for_state_dim

        sd = {
            "proprio.norm.weight": torch.arange(1, 10, dtype=torch.float32),
            "proprio.norm.bias": torch.arange(11, 20, dtype=torch.float32),
            "proprio.proj.weight": torch.arange(18, dtype=torch.float32).view(2, 9),
        }

        adapted = adapt_state_dict_for_state_dim(sd, saved_state_dim=9, current_state_dim=10)

        self.assertEqual(adapted["proprio.norm.weight"].shape, (10,))
        self.assertEqual(adapted["proprio.norm.weight"][:8].tolist(), sd["proprio.norm.weight"][:8].tolist())
        self.assertEqual(float(adapted["proprio.norm.weight"][8]), float(sd["proprio.norm.weight"][8]))
        self.assertEqual(float(adapted["proprio.norm.weight"][9]), float(sd["proprio.norm.weight"][8]))
        self.assertTrue(torch.equal(adapted["proprio.proj.weight"][:, :8], sd["proprio.proj.weight"][:, :8]))
        self.assertTrue(torch.equal(adapted["proprio.proj.weight"][:, 8], sd["proprio.proj.weight"][:, 8] * 0.5))
        self.assertTrue(torch.equal(adapted["proprio.proj.weight"][:, 9], sd["proprio.proj.weight"][:, 8] * 0.5))


if __name__ == "__main__":
    unittest.main()
