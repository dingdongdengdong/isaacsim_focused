"""Shared contracts for the EUN drone RGB camera and crack pipeline."""
from __future__ import annotations

import os
import json
from pathlib import Path


DRONE_PRIM = "/World/eun_iris"
DRONE_BODY_PRIM = f"{DRONE_PRIM}/body"
CAMERA_RELATIVE_PATH = "body/EunFpvCamera"
CAMERA_PRIM = f"{DRONE_PRIM}/{CAMERA_RELATIVE_PATH}"
CAMERA_GRAPH_PRIM = f"{CAMERA_PRIM}_pub"
CAMERA_TRANSLATION_M = (0.30, 0.0, -0.12)
# USD cameras look along local -Z. -90 degrees points along body +X; -80
# degrees adds a 10-degree downward inspection pitch.
CAMERA_ROTATE_Y_DEG = -80.0
CAMERA_RESOLUTION = (640, 480)
CAMERA_RGB_TOPIC = "/drone0/camera/rgb"
CAMERA_FRAME_ID = "drone0_front_lower_camera"
# The camera uses the proven 4.2 m front inspection eye/target. The vehicle root
# starts 0.30 m behind and 0.12 m above the camera so the physical mount remains
# front-lower instead of placing the body origin at the optical center.
CAMERA_INSPECTION_EYE = (20.0, -1.795, 20.75)
CAMERA_INSPECTION_TARGET = (20.0, -5.995, 20.0)
DRONE_INSPECTION_SPAWN_POSITION = (20.0, -1.495, 20.87)
DRONE_INSPECTION_SPAWN_ORIENTATION_XYZW = (0.0, 0.0, -0.70710678, 0.70710678)


SUPPORTED_ENCODINGS = {"rgb8", "bgr8", "rgba8", "bgra8"}


def ros_image_to_rgb_bytes(
    data: bytes,
    *,
    width: int,
    height: int,
    step: int,
    encoding: str,
) -> bytes:
    """Convert a raw ROS ``sensor_msgs/Image`` payload to packed RGB bytes."""
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    normalized_encoding = encoding.lower()
    if normalized_encoding not in SUPPORTED_ENCODINGS:
        raise ValueError(f"unsupported RGB camera encoding: {encoding}")
    channels = 4 if "a" in normalized_encoding else 3
    row_bytes = width * channels
    if step < row_bytes:
        raise ValueError(f"image step {step} is smaller than packed row {row_bytes}")
    if len(data) < step * height:
        raise ValueError("image payload is shorter than step * height")

    source = memoryview(data)
    rgb = bytearray(width * height * 3)
    destination_offset = 0
    source_is_bgr = normalized_encoding.startswith("bgr")
    for row_index in range(height):
        row = source[row_index * step : row_index * step + row_bytes]
        for pixel_offset in range(0, row_bytes, channels):
            first, green, third = row[pixel_offset : pixel_offset + 3]
            if source_is_bgr:
                red, blue = third, first
            else:
                red, blue = first, third
            rgb[destination_offset : destination_offset + 3] = bytes((red, green, blue))
            destination_offset += 3
    return bytes(rgb)


def write_ppm(path: Path, *, width: int, height: int, rgb: bytes) -> None:
    """Atomically write a dependency-free P6 PPM frame."""
    expected_size = width * height * 3
    if len(rgb) != expected_size:
        raise ValueError(f"expected {expected_size} RGB bytes, received {len(rgb)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as stream:
        stream.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        stream.write(rgb)
    os.replace(temporary_path, path)


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON without exposing a partially written status or manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)
