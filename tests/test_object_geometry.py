import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from object_geometry import estimate_object


class GeometryTests(unittest.TestCase):
    def test_center_ray_intersects_known_plane(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[46:55, 46:55] = (255, 0, 0)
        result = estimate_object(image, [0, 0, 1], np.eye(3), 90, plane_z=0)
        np.testing.assert_allclose(result["world_xy_m"], [0, 0], atol=1e-10)

    def test_partial_multiple_and_border_regions_abstain(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[20:24, 20:26] = (255, 0, 0)
        self.assertIsNone(estimate_object(image, [0, 0, 1], np.eye(3), 90))
        image[40:49, 40:49] = (255, 0, 0)
        self.assertIsNone(estimate_object(image, [0, 0, 1], np.eye(3), 90))
        image[:] = 0
        image[0:9, 40:49] = (255, 0, 0)
        self.assertIsNone(estimate_object(image, [0, 0, 1], np.eye(3), 90))

    def test_invalid_camera_calibration_abstains(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[46:55, 46:55] = (255, 0, 0)
        self.assertIsNone(estimate_object(image, [0, 0, float('nan')], np.eye(3), 90))
        self.assertIsNone(estimate_object(image, [0, 0, 1], np.eye(3), 0))
