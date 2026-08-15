#!/usr/bin/env python3
"""Build a two-class YOLO segmentation dataset from the official DACL10K v2 ZIP."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


CLASS_IDS = {"Crack": 0, "Rust": 1}
SPLIT_NAMES = {"train": "train", "validation": "val"}


def normalized_polygon(points: list[list[float]], width: int, height: int) -> list[float] | None:
    if width <= 0 or height <= 0 or len(points) < 3:
        return None
    values: list[float] = []
    unique: set[tuple[float, float]] = set()
    for point in points:
        if len(point) != 2:
            return None
        x = min(max(float(point[0]) / width, 0.0), 1.0)
        y = min(max(float(point[1]) / height, 0.0), 1.0)
        unique.add((x, y))
        values.extend((x, y))
    return values if len(unique) >= 3 else None


def prepare(archive: Path, output: Path) -> dict[str, object]:
    counts: Counter[str] = Counter()
    output.mkdir(parents=True, exist_ok=True)

    with ZipFile(archive) as source:
        annotation_names = sorted(
            name
            for name in source.namelist()
            if "/annotations/" in name and name.endswith(".json")
        )
        if not annotation_names:
            raise ValueError(f"No DACL10K annotations found in {archive}")

        for annotation_name in annotation_names:
            parts = PurePosixPath(annotation_name).parts
            source_split = parts[-2]
            if source_split not in SPLIT_NAMES:
                continue
            split = SPLIT_NAMES[source_split]
            annotation = json.loads(source.read(annotation_name))
            image_name = annotation["imageName"]
            image_member = f"{parts[0]}/images/{source_split}/{image_name}"
            image_output = output / split / "images" / image_name
            label_output = output / split / "labels" / f"{Path(image_name).stem}.txt"
            image_output.parent.mkdir(parents=True, exist_ok=True)
            label_output.parent.mkdir(parents=True, exist_ok=True)

            with source.open(image_member) as src, image_output.open("wb") as dst:
                shutil.copyfileobj(src, dst)

            width = int(annotation["imageWidth"])
            height = int(annotation["imageHeight"])
            labels: list[str] = []
            image_classes: set[str] = set()
            for shape in annotation.get("shapes", []):
                label = shape.get("label")
                if label not in CLASS_IDS:
                    continue
                polygon = normalized_polygon(shape.get("points", []), width, height)
                if polygon is None:
                    counts[f"dropped_{label.lower()}_polygons"] += 1
                    continue
                coords = " ".join(f"{value:.8f}" for value in polygon)
                labels.append(f"{CLASS_IDS[label]} {coords}")
                counts[f"{label.lower()}_polygons"] += 1
                image_classes.add(label)

            label_output.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
            counts[f"{split}_images"] += 1
            if not labels:
                counts[f"{split}_negative_images"] += 1
            for label in image_classes:
                counts[f"{split}_{label.lower()}_images"] += 1

    yaml_text = (
        f"path: {output.resolve()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n"
        "  0: crack\n"
        "  1: rust\n"
    )
    (output / "dacl10k.yaml").write_text(yaml_text, encoding="utf-8")

    report: dict[str, object] = {
        "source_archive": str(archive.resolve()),
        "output": str(output.resolve()),
        "classes": {"0": "crack", "1": "rust"},
        "counts": dict(sorted(counts.items())),
    }
    (output / "preparation-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
