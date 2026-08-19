from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera_pipeline import (  # noqa: E402
    CAMERA_PRIM,
    CAMERA_INSPECTION_EYE,
    CAMERA_INSPECTION_TARGET,
    CAMERA_ROTATE_Y_DEG,
    CAMERA_TRANSLATION_M,
    DRONE_INSPECTION_SPAWN_ORIENTATION_XYZW,
    DRONE_INSPECTION_SPAWN_POSITION,
    ros_image_to_rgb_bytes,
    write_json_atomic,
    write_ppm,
)


class CameraPipelineTests(unittest.TestCase):
    def test_camera_is_front_lower_body_child_with_downward_pitch(self) -> None:
        self.assertTrue(CAMERA_PRIM.startswith("/World/eun_iris/body/"))
        self.assertGreater(CAMERA_TRANSLATION_M[0], 0.0)
        self.assertLess(CAMERA_TRANSLATION_M[2], 0.0)
        self.assertGreater(CAMERA_ROTATE_Y_DEG, -90.0)
        self.assertLess(CAMERA_ROTATE_Y_DEG, -70.0)

    def test_rgb8_with_row_padding_is_packed(self) -> None:
        payload = bytes((1, 2, 3, 4, 5, 6, 99, 99))
        self.assertEqual(
            ros_image_to_rgb_bytes(
                payload,
                width=2,
                height=1,
                step=8,
                encoding="rgb8",
            ),
            bytes((1, 2, 3, 4, 5, 6)),
        )

    def test_spawn_pose_uses_proven_front_inspection_geometry(self) -> None:
        self.assertEqual(CAMERA_INSPECTION_EYE, (20.0, -1.795, 20.75))
        self.assertEqual(CAMERA_INSPECTION_TARGET, (20.0, -5.995, 20.0))
        self.assertEqual(DRONE_INSPECTION_SPAWN_POSITION[:2], (20.0, -1.495))
        self.assertAlmostEqual(DRONE_INSPECTION_SPAWN_POSITION[2], 20.87)
        self.assertAlmostEqual(DRONE_INSPECTION_SPAWN_ORIENTATION_XYZW[2], -0.70710678)
        self.assertAlmostEqual(DRONE_INSPECTION_SPAWN_ORIENTATION_XYZW[3], 0.70710678)

    def test_scene_uses_project_owned_rgb_camera_graph(self) -> None:
        scene_source = (Path(__file__).resolve().parents[1] / "eun_scene.py").read_text()
        self.assertIn("EUNROS2RGBCameraGraph", scene_source)
        graph_source = (
            Path(__file__).resolve().parents[1] / "ros2_rgb_camera_graph.py"
        ).read_text()
        self.assertIn('rep.writers.get("LdrColorSDROS2PublishImage")', graph_source)

    def test_webrtc_defaults_to_drone_camera(self) -> None:
        controller_source = (
            Path(__file__).resolve().parents[1] / "ros2_flight_controller.py"
        ).read_text()
        self.assertIn("self._using_drone_camera = True", controller_source)
        self.assertIn("self._viewport.camera_path = self._drone_camera_path", controller_source)

    def test_camera_mount_has_visible_fixed_housing(self) -> None:
        scene_source = (Path(__file__).resolve().parents[1] / "eun_scene.py").read_text()
        self.assertIn('housing_path = f"{path}/Housing"', scene_source)
        self.assertIn("UsdGeom.Cube.Define(stage, housing_path)", scene_source)

    def test_bgra8_is_converted_to_rgb_without_alpha(self) -> None:
        self.assertEqual(
            ros_image_to_rgb_bytes(
                bytes((10, 20, 30, 255)),
                width=1,
                height=1,
                step=4,
                encoding="bgra8",
            ),
            bytes((30, 20, 10)),
        )

    def test_ppm_writer_is_atomic_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "frame.ppm"
            write_ppm(output, width=1, height=1, rgb=bytes((7, 8, 9)))
            self.assertEqual(output.read_bytes(), b"P6\n1 1\n255\n\x07\x08\x09")
            self.assertFalse(output.with_suffix(".ppm.tmp").exists())

    def test_json_writer_is_atomic_and_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "status.json"
            write_json_atomic(output, {"label": "균열", "ready": True})
            self.assertIn('"ready": true', output.read_text(encoding="utf-8"))
            self.assertIn("균열", output.read_text(encoding="utf-8"))
            self.assertFalse(output.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
