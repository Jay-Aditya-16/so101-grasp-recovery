import sys
import unittest
from pathlib import Path
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from live_act_eval import drawer_success_step, object_inside_box_xy, placement_frame_supported


class PlacementScoringTests(unittest.TestCase):
    def test_full_footprint_must_fit_inside_box(self):
        center = [0.05, 0.22]
        self.assertTrue(object_inside_box_xy(mujoco.mjtGeom.mjGEOM_BOX, [0.025, 0.012, 0.015], np.eye(3), center, center))
        self.assertFalse(object_inside_box_xy(mujoco.mjtGeom.mjGEOM_BOX, [0.025, 0.012, 0.015], np.eye(3), [0.075, 0.22], center))
        self.assertFalse(object_inside_box_xy(mujoco.mjtGeom.mjGEOM_SPHERE, [0.018, 0, 0], np.eye(3), [0.08, 0.22], center))
        rotated = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        self.assertTrue(object_inside_box_xy(mujoco.mjtGeom.mjGEOM_BOX, [0.025, 0.012, 0.015], rotated, [0.075, 0.22], center))
    def test_drawer_requires_sustained_open_distance(self):
        count = 0
        for _ in range(9):
            count, success = drawer_success_step(0.06, count, 20)
            self.assertFalse(success)
        count, success = drawer_success_step(0.059, count, 20)
        self.assertEqual(count, 0)
        self.assertFalse(success)
        for _ in range(10):
            count, success = drawer_success_step(0.06, count, 20)
        self.assertTrue(success)
    def test_requires_support_release_centrality_and_low_speed(self):
        self.assertTrue(placement_frame_supported(True, False, [0.01, 0.02], 0.01))
        self.assertFalse(placement_frame_supported(False, False, [0, 0], 0))
        self.assertFalse(placement_frame_supported(True, True, [0, 0], 0))
        self.assertFalse(placement_frame_supported(True, False, [0.031, 0], 0))
        self.assertFalse(placement_frame_supported(True, False, [0, 0], 0.03))


def test_grasp_friction_rejects_invalid_values():
    from live_act_eval import set_sim_grasp_friction
    import pytest
    for value in (0, float('nan'), 2):
        with pytest.raises(ValueError):
            set_sim_grasp_friction(None, value)


def test_grasp_friction_changes_both_jaws_and_cube_only(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    import live_act_eval
    ids = {'cube_geom': 3, 'gripper': 10, 'moving_jaw_so101_v1': 11}
    monkeypatch.setattr(live_act_eval.mujoco, 'mj_name2id', lambda model, kind, name: ids[name])
    model = SimpleNamespace(ngeom=4, geom_bodyid=np.array([10,11,12,13]), geom_friction=np.ones((4,3)))
    live_act_eval.set_sim_grasp_friction(model, .05)
    assert np.allclose(model.geom_friction[:,0], [.05,.05,1,.05])
    assert np.allclose(model.geom_friction[:,1:], 1)
