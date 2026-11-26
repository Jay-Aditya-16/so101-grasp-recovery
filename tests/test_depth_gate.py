import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from depth_gate import relative_depth_decision
from phase2_controller import RecoveryController


class DepthGateTests(unittest.TestCase):
    def test_closer_and_farther_cube(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[40:50, 40:50] = (255, 0, 0)
        depth = np.zeros((100, 100), dtype=np.float32)
        depth[40:50, 40:50] = 1
        self.assertTrue(relative_depth_decision(rgb, depth)["accepted"])
        depth[40:50, 40:50] = -1
        self.assertFalse(relative_depth_decision(rgb, depth)["accepted"])

    def test_no_object_and_invalid_depth_abstain(self):
        rgb = np.zeros((20, 20, 3), dtype=np.uint8)
        self.assertFalse(relative_depth_decision(rgb, np.zeros((20, 20)))["accepted"])
        self.assertFalse(relative_depth_decision(rgb, np.full((20, 20), np.nan))["accepted"])

    def test_wrist_depth_rejection_stops_descent(self):
        camera = SimpleNamespace(id=0)
        model = SimpleNamespace(camera=lambda name: camera, cam_fovy=np.array([45.0]))
        data = SimpleNamespace(cam_xpos=np.zeros((1, 3)), cam_xmat=np.zeros((1, 9)),
                               ctrl=np.zeros(6), time=1.0)
        robot = SimpleNamespace(model=model, data=data)
        controller = RecoveryController(robot, depth_gate=lambda image: {"accepted": False, "reason": "depth_ambiguous"}, depth_view="wrist")
        controller.state = "descend"
        controller.xy = np.array([0.1, 0.2])
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch("phase2_controller.estimate_object", return_value={"world_xy_m": [0.1, 0.2]}):
            controller.step(rgb, rgb)
        self.assertEqual(controller.state, "aborted")
        self.assertEqual(len(controller.depth_checks), 1)
