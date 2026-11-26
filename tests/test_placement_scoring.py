import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from live_act_eval import placement_frame_supported


class PlacementScoringTests(unittest.TestCase):
    def test_requires_support_release_centrality_and_low_speed(self):
        self.assertTrue(placement_frame_supported(True, False, [0.01, 0.02], 0.01))
        self.assertFalse(placement_frame_supported(False, False, [0, 0], 0))
        self.assertFalse(placement_frame_supported(True, True, [0, 0], 0))
        self.assertFalse(placement_frame_supported(True, False, [0.031, 0], 0))
        self.assertFalse(placement_frame_supported(True, False, [0, 0], 0.03))
