import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class BowlPlacementTest(unittest.TestCase):
    def _fake_name2id(self, _model, _obj_type, name):
        return {
            "cube": 0,
            "target_zone": 1,
            "cube_joint": 2,
            "cube_geom": 3,
            "static_finger_pad": 4,
            "moving_finger_pad": 5,
        }[name]

    def _state(self, cube_xyz, *, qvel=None, contacts=()):
        model = SimpleNamespace(jnt_dofadr=np.array([0, 0, 6], dtype=np.int32))
        xpos = np.zeros((6, 3), dtype=np.float64)
        xpos[0] = cube_xyz
        xpos[1] = [0.32, 0.16, 0.017]
        data = SimpleNamespace(
            xpos=xpos,
            qvel=np.zeros(12, dtype=np.float64) if qvel is None else qvel,
            ncon=len(contacts),
            contact=[SimpleNamespace(geom1=a, geom2=b) for a, b in contacts],
        )
        return model, data

    def test_held_cube_above_bowl_is_not_success(self):
        from bude_vla.envs.so101_mjx import bowl_placement_state

        model, data = self._state(
            [0.32, 0.16, 0.09], contacts=[(3, 4), (3, 5)]
        )
        with patch("bude_vla.envs.so101_mjx.mujoco.mj_name2id", self._fake_name2id):
            state = bowl_placement_state(model, data)

        self.assertTrue(state.inside_xy)
        self.assertFalse(state.low_enough)
        self.assertFalse(state.released)
        self.assertFalse(state.placed)

    def test_released_still_cube_inside_bowl_is_success(self):
        from bude_vla.envs.so101_mjx import bowl_placement_state

        model, data = self._state([0.329, 0.16, 0.035])
        with patch("bude_vla.envs.so101_mjx.mujoco.mj_name2id", self._fake_name2id):
            state = bowl_placement_state(model, data)

        self.assertTrue(state.inside_xy)
        self.assertTrue(state.low_enough)
        self.assertTrue(state.released)
        self.assertTrue(state.settled)
        self.assertTrue(state.placed)

    def test_tracker_requires_consecutive_stable_frames(self):
        from bude_vla.envs.so101_mjx import BowlPlacementTracker

        model, data = self._state([0.32, 0.16, 0.035])
        tracker = BowlPlacementTracker(required_steps=3)
        with patch("bude_vla.envs.so101_mjx.mujoco.mj_name2id", self._fake_name2id):
            self.assertFalse(tracker.update(model, data))
            self.assertFalse(tracker.update(model, data))
            data.xpos[0, 2] = 0.08
            self.assertFalse(tracker.update(model, data))
            data.xpos[0, 2] = 0.035
            self.assertFalse(tracker.update(model, data))
            self.assertFalse(tracker.update(model, data))
            self.assertTrue(tracker.update(model, data))


if __name__ == "__main__":
    unittest.main()
