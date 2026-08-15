from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steelcrack_usd import minimum_camera_distance_for_decal  # noqa: E402


class SceneContractTests(unittest.TestCase):
    def test_camera_distance_keeps_rotated_decal_inside_frame(self) -> None:
        distance = minimum_camera_distance_for_decal((2.4, 1.3), 18.0)
        self.assertGreater(distance, 4.0)
        self.assertLess(distance, 5.0)

    def test_cube_crack_is_replaced_by_rgba_decal_contract(self) -> None:
        scene = (Path(__file__).resolve().parents[1] / "eun_scene.py").read_text(encoding="utf-8")
        self.assertNotIn("add_crack_segment", scene)
        self.assertNotIn("Outline_", scene)
        self.assertNotIn("Core_", scene)
        for key in (
            '"crack_mode"',
            '"source_dataset"',
            '"source_split"',
            '"source_id"',
            '"texture_sha256"',
            '"decal_prim"',
            '"inspection_camera"',
            '"semantic_label"',
        ):
            self.assertIn(key, scene)

    def test_usd_helper_has_expected_prim_material_and_pose(self) -> None:
        helper = (Path(__file__).resolve().parents[1] / "steelcrack_usd.py").read_text(encoding="utf-8")
        self.assertIn('/World/TransferCrane/CrackDecals', helper)
        self.assertIn('SteelCrack_000053', helper)
        self.assertIn('UsdPreviewSurface', helper)
        self.assertIn('UsdUVTexture', helper)
        self.assertIn('opacityThreshold', helper)
        self.assertIn('position=(0.0, 20.0, -4.005)', helper)
        self.assertIn('size=(2.0, 1.5)', helper)


if __name__ == "__main__":
    unittest.main()
