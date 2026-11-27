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


class GraspCandidateTests(unittest.TestCase):
    def test_cube_contact_heights_are_named_and_preserve_xy(self):
        import numpy as np
        for name, z in (("low", .015), ("center", .020), ("high", .025)):
            robot = SimpleNamespace(data=SimpleNamespace(ctrl=np.zeros(6)))
            controller = RecoveryController(robot, grasp_candidate=name)
            controller.state = 'descend'
            controller.xy = np.array([.2, -.03])
            calls = []
            controller.target = lambda *args, **kwargs: calls.append((args, kwargs))
            controller.step(np.zeros((1,1,3)))
            self.assertEqual(calls[0][0][0], [.2, -.03, z])
            self.assertEqual(calls[0][0][1], 'close')
            self.assertTrue(calls[0][1]['allow_cube_contact'])

    def test_unknown_candidate_rejected(self):
        with self.assertRaises(ValueError):
            RecoveryController(SimpleNamespace(), grasp_candidate='unrecognized')
