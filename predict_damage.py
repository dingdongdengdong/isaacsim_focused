#!/usr/bin/env python3
"""Run crack/rust segmentation with fixed, easy-to-read class colors."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


COLORS = {
    0: (0, 0, 255),  # crack: red (BGR)
    1: (255, 0, 0),  # rust: blue (BGR)
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    image = cv2.imread(str(args.source))
    if image is None:
        raise ValueError(f"Could not read image: {args.source}")

    result = YOLO(args.model).predict(
        source=str(args.source),
        imgsz=640,
        conf=args.confidence,
        device=args.device,
        verbose=False,
    )[0]
    rendered = image.copy()
    detections: list[str] = []

    if result.masks is not None and result.boxes is not None:
        for mask_tensor, box in zip(result.masks.data, result.boxes):
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            color = COLORS.get(class_id, (0, 255, 255))
            mask = mask_tensor.cpu().numpy()
            mask = cv2.resize(mask, (image.shape[1], image.shape[0])) > 0.5

            colored = np.full_like(rendered, color)
            rendered[mask] = cv2.addWeighted(
                rendered, 0.35, colored, 0.65, 0
            )[mask]
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(rendered, contours, -1, color, 2)

            x1, y1, _, _ = (int(value) for value in box.xyxy[0].tolist())
            label = f"{result.names[class_id]} {confidence:.2f}"
            cv2.putText(
                rendered,
                label,
                (x1, max(y1 - 7, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
            detections.append(label)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), rendered):
        raise RuntimeError(f"Could not write result: {args.output}")
    print(f"detections={len(detections)}")
    for detection in detections:
        print(detection)
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
