import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scenario_suite import scenario_scene


class ScenarioTests(unittest.TestCase):
    def test_template_is_restored_after_exception(self):
        original = 'type="box" size="0.015 0.015 0.015" {cube_y} 0.015'
        scene = SimpleNamespace(SCENE_TEMPLATE=original)
        with patch.dict(sys.modules, {"common": SimpleNamespace(scene=scene)}):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                with scenario_scene("small_cube"):
                    self.assertIn('size="0.010 0.010 0.010"', scene.SCENE_TEMPLATE)
                    self.assertIn('{cube_y} 0.010', scene.SCENE_TEMPLATE)
                    raise RuntimeError("stop")
        self.assertEqual(scene.SCENE_TEMPLATE, original)

    def test_unknown_case_is_rejected(self):
        with self.assertRaises(ValueError):
            with scenario_scene("unrecognized"):
                pass
