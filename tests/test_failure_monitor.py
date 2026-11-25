import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from failure_monitor import GraspMonitor


def frame(x=None):
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    if x is not None:
        image[30:40, x:x + 10] = [255, 0, 0]
    return image


class GraspMonitorTests(unittest.TestCase):
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
