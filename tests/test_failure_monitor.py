import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from failure_monitor import GraspMonitor, empty_closed_gripper, stationary_cube_after_unknown


def frame(x=None):
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    if x is not None:
        image[30:40, x:x + 10] = [255, 0, 0]
    return image


class GraspMonitorTests(unittest.TestCase):
    def test_stationary_fallback_requires_visible_object_and_delay(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[46:55, 46:55] = [255, 0, 0]
        self.assertFalse(stationary_cube_after_unknown(image, [50, 50], 9, 20))
        self.assertTrue(stationary_cube_after_unknown(image, [50, 50], 10, 20))
        self.assertFalse(stationary_cube_after_unknown(image, [10, 10], 10, 20))
        self.assertFalse(stationary_cube_after_unknown(np.zeros_like(image), [50, 50], 10, 20))
    def test_empty_closure_requires_finite_closed_target_and_position(self):
        self.assertTrue(empty_closed_gripper(-0.1, -0.08))
        self.assertFalse(empty_closed_gripper(-0.1, 0.147))
        self.assertFalse(empty_closed_gripper(1.2, -0.08))
        self.assertFalse(empty_closed_gripper(-0.1, float("nan")))
    def test_stationary_cube_after_lift_is_suspected_miss(self):
        monitor = GraspMonitor()
        self.assertIsNone(monitor.update(frame(30), 1.0, 0.10))
        self.assertIsNone(monitor.update(frame(30), 0.0, 0.10))
        event = monitor.update(frame(31), 0.0, 0.13)
        self.assertEqual(event.status, "suspected_miss")

    def test_occlusion_and_moved_cube_are_unknown(self):
        for observed in (frame(), frame(60)):
            monitor = GraspMonitor()
            monitor.update(frame(30), 1.0, 0.10)
            monitor.update(frame(30), 0.0, 0.10)
            self.assertEqual(monitor.update(observed, 0.0, 0.13).status, "unknown")

    def test_initial_closed_and_no_preclosure_image_is_unknown(self):
        monitor = GraspMonitor()
        monitor.update(frame(), 0.0, 0.10)
        self.assertEqual(monitor.update(frame(30), 0.0, 0.13).status, "unknown")

    def test_stale_object_detection_is_unknown(self):
        monitor = GraspMonitor(max_age_frames=2)
        monitor.update(frame(30), 1.0, 0.10)
        for _ in range(3):
            monitor.update(frame(), 1.0, 0.10)
        monitor.update(frame(), 0.0, 0.10)
        self.assertEqual(monitor.update(frame(30), 0.0, 0.13).status, "unknown")


if __name__ == "__main__":
    unittest.main()
