from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detect_crack_stream import (  # noqa: E402
    MIN_CRACK_COMPONENT_PIXELS,
    component_geometry_is_plausible_crack,
    frame_paths,
)


class DetectCrackStreamTests(unittest.TestCase):
    def test_minimum_component_removes_edge_speckles(self) -> None:
        self.assertGreater(MIN_CRACK_COMPONENT_PIXELS, 33)
        self.assertLess(MIN_CRACK_COMPONENT_PIXELS, 1323)

    def test_frame_paths_are_ordered_and_ignore_partial_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frame-000002.ppm").touch()
            (root / "frame-000001.ppm").touch()
            (root / "frame-000003.ppm.tmp").touch()
            self.assertEqual(
                [path.name for path in frame_paths(root)],
                ["frame-000001.ppm", "frame-000002.ppm"],
            )

    def test_rejects_thin_full_width_crane_rail(self) -> None:
        self.assertFalse(
            component_geometry_is_plausible_crack(
                width=640,
                height=12,
                image_width=640,
                image_height=480,
            )
        )

    def test_accepts_vertical_crack_component(self) -> None:
        self.assertTrue(
            component_geometry_is_plausible_crack(
                width=82,
                height=180,
                image_width=640,
                image_height=480,
            )
        )

    def test_accepts_non_extreme_diagonal_crack_component(self) -> None:
        self.assertTrue(
            component_geometry_is_plausible_crack(
                width=160,
                height=55,
                image_width=640,
                image_height=480,
            )
        )


if __name__ == "__main__":
    unittest.main()
