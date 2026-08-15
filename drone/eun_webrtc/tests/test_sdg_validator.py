from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate_steelcrack_sdg import perspective_coefficients, project_alpha, project_point  # noqa: E402


class SDGValidatorTests(unittest.TestCase):
    def test_identity_perspective_projection_preserves_mask(self) -> None:
        alpha = Image.new("L", (8, 8), 0)
        for coordinate in ((2, 2), (3, 2), (3, 3)):
            alpha.putpixel(coordinate, 255)
        quad = [(0.0, 7.0), (7.0, 7.0), (7.0, 0.0), (0.0, 0.0)]
        projected = project_alpha(alpha, quad, (8, 8))
        self.assertEqual(sum(1 for value in projected.getdata() if value), 3)

    def test_perspective_coefficients_map_corners(self) -> None:
        destination = [(10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)]
        source = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        coefficients = perspective_coefficients(destination, source)
        self.assertEqual(len(coefficients), 8)

    def test_camera_projection_places_target_at_image_center(self) -> None:
        pose = {"eye": [0.0, 4.0, 0.0], "target": [0.0, 0.0, 0.0]}
        intrinsics = {
            "focal_length_mm": 35.0,
            "horizontal_aperture_mm": 20.0,
            "vertical_aperture_mm": 20.0,
            "resolution": [512, 512],
        }
        x, y = project_point([0.0, 0.0, 0.0], pose, intrinsics)
        self.assertAlmostEqual(x, 256.0)
        self.assertAlmostEqual(y, 256.0)


if __name__ == "__main__":
    unittest.main()
