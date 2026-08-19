#!/usr/bin/env python3
"""Capture Isaac Sim ROS2 RGB frames without cv_bridge or Replicator."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from camera_pipeline import (
    CAMERA_RGB_TOPIC,
    ros_image_to_rgb_bytes,
    write_json_atomic,
    write_ppm,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topic", default=CAMERA_RGB_TOPIC)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--frame-stride", type=int, default=6)
    args = parser.parse_args()
    if args.max_frames <= 0 or args.frame_stride <= 0:
        parser.error("--max-frames and --frame-stride must be positive")

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "capture_status.json"
    started_at = time.time()

    class RGBFrameCapture(Node):
        def __init__(self) -> None:
            super().__init__("eun_rgb_frame_capture")
            self.received = 0
            self.saved = 0
            self.width = 0
            self.height = 0
            self.encoding = ""
            self.create_subscription(Image, args.topic, self.on_image, qos_profile_sensor_data)

        def on_image(self, message: Image) -> None:
            self.received += 1
            if (self.received - 1) % args.frame_stride:
                return
            rgb = ros_image_to_rgb_bytes(
                bytes(message.data),
                width=message.width,
                height=message.height,
                step=message.step,
                encoding=message.encoding,
            )
            self.saved += 1
            self.width = int(message.width)
            self.height = int(message.height)
            self.encoding = message.encoding
            frame_path = args.output_dir / f"frame-{self.saved:06d}.ppm"
            write_ppm(frame_path, width=self.width, height=self.height, rgb=rgb)
            write_json_atomic(
                status_path,
                {
                    "ready": self.saved >= args.max_frames,
                    "topic": args.topic,
                    "received_messages": self.received,
                    "saved_frames": self.saved,
                    "width": self.width,
                    "height": self.height,
                    "encoding": self.encoding,
                    "elapsed_seconds": time.time() - started_at,
                    "latest_frame": str(frame_path),
                },
            )
            if self.saved >= args.max_frames:
                rclpy.shutdown()

    rclpy.init()
    node = RGBFrameCapture()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
