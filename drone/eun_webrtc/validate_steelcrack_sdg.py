#!/usr/bin/env python3
"""Validate SteelCrack SDG output and derive homography-projected masks."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter


def _dot(left, right) -> float:
    return sum(a * b for a, b in zip(left, right))


def _sub(left, right) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(left, right))


def _cross(left, right) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _normalise(vector) -> tuple[float, float, float]:
    norm = math.sqrt(_dot(vector, vector))
    if norm == 0:
        raise ValueError("zero-length camera vector")
    return tuple(value / norm for value in vector)


def project_point(point, camera_pose: dict, camera_intrinsics: dict) -> tuple[float, float]:
    eye = camera_pose["eye"]
    target = camera_pose["target"]
    forward = _normalise(_sub(target, eye))
    right = _normalise(_cross(forward, (0.0, 0.0, 1.0)))
    camera_up = _normalise(_cross(right, forward))
    relative = _sub(point, eye)
    depth = _dot(relative, forward)
    if depth <= 0:
        raise ValueError("decal corner is behind the camera")
    camera_x = _dot(relative, right)
    camera_y = _dot(relative, camera_up)
    focal = camera_intrinsics["focal_length_mm"]
    horizontal_aperture = camera_intrinsics["horizontal_aperture_mm"]
    vertical_aperture = camera_intrinsics["vertical_aperture_mm"]
    width, height = camera_intrinsics["resolution"]
    pixel_x = width * (0.5 + focal * camera_x / (horizontal_aperture * depth))
    pixel_y = height * (0.5 - focal * camera_y / (vertical_aperture * depth))
    return pixel_x, pixel_y


def _solve_linear(matrix: list[list[float]], values: list[float]) -> list[float]:
    size = len(values)
    augmented = [row[:] + [value] for row, value in zip(matrix, values)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1.0e-12:
            raise ValueError("singular homography")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def perspective_coefficients(destination, source) -> tuple[float, ...]:
    """Return Pillow output-to-input perspective coefficients."""
    matrix = []
    values = []
    for (x, y), (u, v) in zip(destination, source):
        matrix.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        values.append(u)
        matrix.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.append(v)
    return tuple(_solve_linear(matrix, values))


def project_alpha(alpha: Image.Image, destination_quad, resolution) -> Image.Image:
    source_quad = [
        (0.0, float(alpha.height - 1)),
        (float(alpha.width - 1), float(alpha.height - 1)),
        (float(alpha.width - 1), 0.0),
        (0.0, 0.0),
    ]
    coefficients = perspective_coefficients(destination_quad, source_quad)
    binary = alpha.point(lambda value: 255 if value >= 26 else 0)
    perspective_mode = getattr(getattr(Image, "Transform", Image), "PERSPECTIVE", Image.PERSPECTIVE)
    nearest_mode = getattr(getattr(Image, "Resampling", Image), "NEAREST", Image.NEAREST)
    return binary.transform(
        tuple(resolution),
        perspective_mode,
        coefficients,
        resample=nearest_mode,
        fillcolor=0,
    )


def _bbox(mask: Image.Image, predicate=lambda value: value != 0):
    if predicate(255) and not predicate(0):
        return mask.getbbox()
    width, height = mask.size
    selected = [index for index, value in enumerate(mask.getdata()) if predicate(value)]
    if not selected:
        return None
    xs = [index % width for index in selected]
    ys = [index // width for index in selected]
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def _bbox_iou(left, right) -> float:
    if left is None or right is None:
        return 0.0
    intersection_width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _quad_bbox(quad, resolution):
    width, height = resolution
    return (
        max(0, math.floor(min(point[0] for point in quad))),
        max(0, math.floor(min(point[1] for point in quad))),
        min(width, math.ceil(max(point[0] for point in quad))),
        min(height, math.ceil(max(point[1] for point in quad))),
    )


def _semantic_id(labels: dict, semantic_label: str):
    for key, value in labels.items():
        if isinstance(value, dict) and semantic_label in value.values():
            return int(key)
        if semantic_label in str(value):
            return int(key)
    return None


def _semantic_binary(mask_path: Path, labels_path: Path, semantic_label: str) -> Image.Image:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    target_id = _semantic_id(labels, semantic_label)
    with Image.open(mask_path) as image:
        raw = image.copy()
    if target_id is None:
        return Image.new("L", raw.size, 0)
    if not 0 <= target_id <= 255:
        raise ValueError(f"semantic ID {target_id} cannot be represented in the validator's L-mode mask")
    return raw.convert("L").point([255 if value == target_id else 0 for value in range(256)])


def _mask_iou(left: Image.Image, right: Image.Image) -> float:
    left_binary = left.convert("1")
    right_binary = right.convert("1")
    intersection = ImageChops.logical_and(left_binary, right_binary).histogram()[255]
    union = ImageChops.logical_or(left_binary, right_binary).histogram()[255]
    return intersection / union if union else 1.0


def _mask_iou_with_one_pixel_tolerance(left: Image.Image, right: Image.Image) -> float:
    """Compare thin rasterized cracks while allowing one pixel of edge error."""
    return _mask_iou(
        left.convert("L").filter(ImageFilter.MaxFilter(3)),
        right.convert("L").filter(ImageFilter.MaxFilter(3)),
    )


def _contact_sheet(output_root: Path, records: list[dict], columns: int = 4) -> str:
    chosen = records[:12]
    thumb_size = (256, 256)
    rows = math.ceil(len(chosen) / columns)
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * thumb_size[1]), (32, 32, 32))
    for index, record in enumerate(chosen):
        with Image.open(output_root / record["rgb_path"]) as image:
            rgb = image.convert("RGB")
        with Image.open(output_root / record["mask_path"]) as mask_image:
            mask = mask_image.convert("L")
        overlay = Image.new("RGB", rgb.size, (255, 0, 0))
        composite = Image.composite(overlay, rgb, mask.point(lambda value: 120 if value else 0))
        composite.thumbnail(thumb_size)
        draw = ImageDraw.Draw(composite)
        draw.rectangle((0, 0, 150, 18), fill=(0, 0, 0))
        draw.text((4, 3), f"frame {record['frame_id']} source={record['source_id']}", fill=(255, 255, 255))
        x = (index % columns) * thumb_size[0]
        y = (index // columns) * thumb_size[1]
        sheet.paste(composite, (x, y))
    path = output_root / "validation_contact_sheet.png"
    sheet.save(path)
    return str(path)


def validate(output_root: Path, asset_dir: Path) -> dict:
    output_root = output_root.resolve()
    asset_dir = asset_dir.resolve()
    manifest_path = output_root / "manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    selection = json.loads((asset_dir / "selected_sources.json").read_text())
    selected_ids = {source["source_id"] for source in selection["sources"]}
    failures = []
    frame_reports = []
    positive_count = 0
    clean_count = 0
    for record in records:
        frame_id = record["frame_id"]
        rgb_path = output_root / record["rgb_path"]
        semantic_path = output_root / record["semantic_mask_path"]
        labels_path = output_root / record["semantic_labels_path"]
        camera_params_path = output_root / record["camera_params_path"]
        required = (rgb_path, semantic_path, labels_path, camera_params_path)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            failures.append({"frame_id": frame_id, "reason": "missing_writer_output", "paths": missing})
            continue
        with Image.open(rgb_path) as rgb_image:
            rgb_size = rgb_image.size
        resolution = tuple(record["camera_intrinsics"]["resolution"])
        if rgb_size != resolution:
            failures.append(
                {"frame_id": frame_id, "reason": "rgb_size_mismatch", "actual": rgb_size, "expected": resolution}
            )

        clean = record["clean_hard_negative"]
        if clean:
            clean_count += 1
            projected = Image.new("L", resolution, 0)
        else:
            positive_count += 1
            source_id = record["source_id"]
            if record["source_split"] != "Train" or source_id not in selected_ids:
                failures.append({"frame_id": frame_id, "reason": "non_train_or_unselected_source"})
            with Image.open(asset_dir / f"SteelCrack_{source_id}.png") as texture:
                alpha = texture.getchannel("A")
            destination_quad = [
                project_point(point, record["camera_pose"], record["camera_intrinsics"])
                for point in record["decal_world_corners"]
            ]
            projected = project_alpha(alpha, destination_quad, resolution)

        mask_path = output_root / record["mask_path"]
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        projected.save(mask_path)
        projected_bbox = _bbox(projected)
        projected_pixels = sum(projected.histogram()[1:])
        semantic = _semantic_binary(semantic_path, labels_path, record["semantic_label"])
        semantic_bbox = _bbox(semantic)
        semantic_pixels = sum(semantic.histogram()[1:])
        if clean:
            alpha_semantic_iou = 1.0 if semantic_pixels == 0 else 0.0
            tolerant_alpha_semantic_iou = alpha_semantic_iou
            if projected_pixels or semantic_pixels:
                failures.append({"frame_id": frame_id, "reason": "clean_mask_not_empty"})
        else:
            quad = [
                project_point(point, record["camera_pose"], record["camera_intrinsics"])
                for point in record["decal_world_corners"]
            ]
            semantic_quad_bbox_iou = _bbox_iou(semantic_bbox, _quad_bbox(quad, resolution))
            alpha_semantic_iou = _mask_iou(projected, semantic)
            tolerant_alpha_semantic_iou = _mask_iou_with_one_pixel_tolerance(projected, semantic)
            if projected_pixels == 0:
                failures.append({"frame_id": frame_id, "reason": "empty_projected_mask"})
            if semantic_pixels == 0:
                failures.append({"frame_id": frame_id, "reason": "empty_semantic_mask"})
            if tolerant_alpha_semantic_iou < 0.85:
                failures.append(
                    {
                        "frame_id": frame_id,
                        "reason": "rgb_mask_alignment_error",
                        "alpha_semantic_iou": alpha_semantic_iou,
                        "one_pixel_tolerant_iou": tolerant_alpha_semantic_iou,
                        "semantic_quad_bbox_iou": semantic_quad_bbox_iou,
                    }
                )
        frame_reports.append(
            {
                "frame_id": frame_id,
                "source_id": record["source_id"],
                "clean_hard_negative": clean,
                "rgb_size": list(rgb_size),
                "projected_mask_pixels": projected_pixels,
                "semantic_mask_pixels": semantic_pixels,
                "alpha_semantic_iou": alpha_semantic_iou,
                "one_pixel_tolerant_iou": tolerant_alpha_semantic_iou,
            }
        )

    expected_clean = (len(records) + 4) // 5
    if clean_count != expected_clean:
        failures.append(
            {"reason": "clean_ratio_mismatch", "clean": clean_count, "expected": expected_clean}
        )
    contact_sheet = _contact_sheet(output_root, records) if records else None
    report = {
        "passed": not failures,
        "dataset": "SteelCrack crane decal SDG",
        "source_split": "Train",
        "writer": "Isaac Sim 5.1 BasicWriter",
        "mask_method": "source RGBA alpha threshold projected with planar homography",
        "semantic_cross_check": (
            "exact pixel IoU plus one-pixel-tolerant IoU between Replicator semantic mask "
            "and homography-projected source alpha"
        ),
        "frames": len(records),
        "positive_frames": positive_count,
        "clean_hard_negative_frames": clean_count,
        "failures": failures,
        "frame_reports": frame_reports,
        "contact_sheet": contact_sheet,
    }
    (output_root / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    status_path = output_root / "status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    status.update(
        {
            "state": "validated" if report["passed"] else "validation_failed",
            "validation_passed": report["passed"],
            "validation_report": str(output_root / "validation.json"),
            "contact_sheet": contact_sheet,
        }
    )
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=Path("/home/dong/eun/drone/.runtime/eun-webrtc/crack_assets"),
    )
    args = parser.parse_args()
    report = validate(args.output_root, args.asset_dir)
    print(json.dumps({key: value for key, value in report.items() if key != "frame_reports"}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
