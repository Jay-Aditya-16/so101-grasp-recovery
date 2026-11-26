import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from phase2_controller import RecoveryController


class GripperEvidenceTests(unittest.TestCase):
    def test_blocked_gripper_is_plausible(self):
        robot = SimpleNamespace(data=SimpleNamespace(time=1.0), gripper_opening=lambda: 0.147)
        controller = RecoveryController(robot)
        self.assertTrue(controller.check_gripper("after_lift"))
        self.assertTrue(controller.events[-1]["plausible"])

    def test_empty_or_invalid_gripper_aborts(self):
        for value in (-0.1, float("nan")):
            robot = SimpleNamespace(data=SimpleNamespace(time=1.0), gripper_opening=lambda: value)
            controller = RecoveryController(robot)
            self.assertFalse(controller.check_gripper("after_lift"))
            self.assertEqual(controller.state, "aborted")
