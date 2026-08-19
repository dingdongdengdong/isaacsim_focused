#!/usr/bin/env python3
"""Run crack segmentation over captured drone RGB frames."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from camera_pipeline import CAMERA_RGB_TOPIC, write_json_atomic


CRACK_COLOR_BGR = (0, 0, 255)
MIN_CRACK_COMPONENT_PIXELS = 48


def frame_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(source.glob("frame-*.ppm"))


def component_geometry_is_plausible_crack(
    *, width: int, height: int, image_width: int, image_height: int
) -> bool:
    """Exclude thin scene-wide rails while retaining compact crack shapes."""
    if width <= 0 or height <= 0:
        return False
    is_thin_scene_wide_horizontal = (
        width >= 0.75 * image_width
        and height <= 0.08 * image_height
        and width / height >= 8.0
    )
    return not is_thin_scene_wide_horizontal


class SegformerCrackDetector:
    name = "SteelCrack SegFormer-B0"

    def __init__(self, model_path: Path, device: str, threshold: float) -> None:
        import torch
        from transformers import SegformerForSemanticSegmentation

        if device == "cpu":
            self.device = torch.device("cpu")
        else:
            cuda_index = int(device)
            self.device = torch.device(f"cuda:{cuda_index}")
        self.torch = torch
        self.threshold = threshold
        self.input_size = 512
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_path)
        self.model.to(self.device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def predict(self, image) -> list[dict]:
        import cv2
        import numpy as np
        import torch.nn.functional as functional

        height, width = image.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        left = (self.input_size - resized_width) // 2
        top = (self.input_size - resized_height) // 2
        canvas = np.empty((self.input_size, self.input_size, 3), dtype=np.uint8)
        canvas[:] = (124, 116, 104)
        canvas[top : top + resized_height, left : left + resized_width] = resized
        tensor = self.torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device, dtype=self.torch.float32) / 255.0
        tensor = (tensor - self.mean) / self.std
        use_cuda = self.device.type == "cuda"
        with self.torch.inference_mode(), self.torch.autocast(
            device_type=self.device.type,
            dtype=self.torch.float16 if use_cuda else self.torch.bfloat16,
            enabled=use_cuda,
        ):
            logits = self.model(pixel_values=tensor).logits
            logits = functional.interpolate(
                logits,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
            probability_square = self.torch.softmax(logits, dim=1)[0, 1].float().cpu().numpy()
        probability = probability_square[
            top : top + resized_height,
            left : left + resized_width,
        ]
        probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
        raw_mask = probability >= self.threshold
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            raw_mask.astype(np.uint8), 8
        )
        detections = []
        for component in range(1, component_count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            component_width = int(stats[component, cv2.CC_STAT_WIDTH])
            component_height = int(stats[component, cv2.CC_STAT_HEIGHT])
            if area < MIN_CRACK_COMPONENT_PIXELS or not component_geometry_is_plausible_crack(
                width=component_width,
                height=component_height,
                image_width=width,
                image_height=height,
            ):
                continue
            mask = labels == component
            x1 = int(stats[component, cv2.CC_STAT_LEFT])
            y1 = int(stats[component, cv2.CC_STAT_TOP])
            detections.append(
                {
                    "mask": mask,
                    "confidence": float(probability[mask].mean()),
                    "box_xyxy": [
                        x1,
                        y1,
                        x1 + component_width - 1,
                        y1 + component_height - 1,
                    ],
                }
            )
        return detections


class YoloCrackDetector:
    name = "DACL10K YOLO segmentation"

    def __init__(self, model_path: Path, device: str, threshold: float) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.device = device
        self.threshold = threshold

    def predict(self, image) -> list[dict]:
        import cv2

        result = self.model.predict(
            image,
            conf=self.threshold,
            imgsz=640,
            device=self.device,
            verbose=False,
        )[0]
        detections = []
        if result.masks is None or result.boxes is None:
            return detections
        for mask_tensor, box in zip(result.masks.data, result.boxes):
            class_id = int(box.cls.item())
            if str(result.names[class_id]).lower() != "crack":
                continue
            mask = cv2.resize(
                mask_tensor.cpu().numpy(),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            ) > 0.5
            detections.append(
                {
                    "mask": mask,
                    "confidence": float(box.conf.item()),
                    "box_xyxy": [int(value) for value in box.xyxy[0].tolist()],
                }
            )
        return detections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--backend", choices=("segformer", "yolo"), default="segformer")
    parser.add_argument("--device", default="0")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if not args.model.exists():
        parser.error(f"model does not exist: {args.model}")

    import cv2
    import numpy as np

    inputs = frame_paths(args.source)
    if args.max_frames > 0:
        inputs = inputs[: args.max_frames]
    if not inputs:
        raise FileNotFoundError(f"no captured RGB frames found under {args.source}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = args.output_dir / "annotated_frames"
    annotated_dir.mkdir(exist_ok=True)
    detector = (
        SegformerCrackDetector(args.model, args.device, args.confidence)
        if args.backend == "segformer"
        else YoloCrackDetector(args.model, args.device, args.confidence)
    )
    metrics_path = args.output_dir / "detections.jsonl"
    video_path = args.output_dir / "05_drone_to_crack_ai.mp4"
    writer = None
    best = None
    total_crack_detections = 0
    started_at = time.perf_counter()

    try:
        with metrics_path.open("w", encoding="utf-8") as metrics:
            for frame_index, input_path in enumerate(inputs, start=1):
                image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"could not decode camera frame: {input_path}")
                if writer is None:
                    height, width = image.shape[:2]
                    writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        args.fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"could not create video: {video_path}")
                    if not cv2.imwrite(
                        str(args.output_dir / "02_drone_flight_rgb.jpg"), image
                    ):
                        raise RuntimeError("could not write the first RGB evidence frame")

                inference_started = time.perf_counter()
                raw_detections = detector.predict(image)
                inference_ms = (time.perf_counter() - inference_started) * 1000.0
                rendered = image.copy()
                detections = []
                for raw_detection in raw_detections:
                    mask = raw_detection["mask"]
                    confidence = raw_detection["confidence"]
                    x1, y1, x2, y2 = raw_detection["box_xyxy"]
                    colored = np.full_like(rendered, CRACK_COLOR_BGR)
                    rendered[mask] = cv2.addWeighted(
                        rendered, 0.35, colored, 0.65, 0
                    )[mask]
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(rendered, contours, -1, CRACK_COLOR_BGR, 2)
                    cv2.putText(
                        rendered,
                        f"crack {confidence:.2f}",
                        (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        CRACK_COLOR_BGR,
                        2,
                        cv2.LINE_AA,
                    )
                    detections.append(
                        {
                            "class": "crack",
                            "confidence": confidence,
                            "box_xyxy": [x1, y1, x2, y2],
                            "mask_pixels": int(mask.sum()),
                        }
                    )

                total_crack_detections += len(detections)
                header = (
                    f"EUN drone RGB | crack detections {len(detections)} | "
                    f"inference {inference_ms:.1f} ms"
                )
                cv2.rectangle(rendered, (0, 0), (rendered.shape[1], 32), (0, 0, 0), -1)
                cv2.putText(
                    rendered,
                    header,
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                annotated_path = annotated_dir / f"frame-{frame_index:06d}.jpg"
                if not cv2.imwrite(str(annotated_path), rendered):
                    raise RuntimeError(f"could not write annotated frame: {annotated_path}")
                writer.write(rendered)
                record = {
                    "frame": frame_index,
                    "source": str(input_path),
                    "annotated": str(annotated_path),
                    "inference_ms": inference_ms,
                    "detections": detections,
                }
                metrics.write(json.dumps(record, sort_keys=True) + "\n")
                frame_best = max((item["confidence"] for item in detections), default=-1.0)
                if best is None or frame_best > best[0]:
                    best = (frame_best, image.copy(), rendered.copy(), record)
    finally:
        if writer is not None:
            writer.release()

    if best is None:
        raise RuntimeError("no frames were processed")
    if not cv2.imwrite(str(args.output_dir / "03_crane_crack_from_drone.jpg"), best[1]):
        raise RuntimeError("could not write the best crack-view frame")
    if not cv2.imwrite(str(args.output_dir / "04_ai_crack_detection.jpg"), best[2]):
        raise RuntimeError("could not write the best AI result frame")
    manifest = {
        "pipeline": f"Isaac Sim drone -> ROS2 RGB -> {detector.name} -> visualization",
        "backend": args.backend,
        "source_topic": CAMERA_RGB_TOPIC,
        "model": str(args.model.resolve()),
        "confidence_threshold": args.confidence,
        "processed_frames": len(inputs),
        "total_crack_detections": total_crack_detections,
        "best_frame": best[3],
        "elapsed_seconds": time.perf_counter() - started_at,
        "artifacts": {
            "flight_rgb": "02_drone_flight_rgb.jpg",
            "crack_view": "03_crane_crack_from_drone.jpg",
            "ai_result": "04_ai_crack_detection.jpg",
            "video": "05_drone_to_crack_ai.mp4",
        },
    }
    mount_evidence = args.output_dir / "01_rgb_camera_mounted_drone.png"
    if mount_evidence.is_file():
        manifest["artifacts"]["camera_mount"] = mount_evidence.name
    write_json_atomic(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
