#!/usr/bin/env python3
"""Prepare non-redistributable SteelCrack Train decals for the EUN runtime.

The output belongs under ``.runtime/eun-webrtc/crack_assets`` and is not source
material.  Validation and Test are never read by this tool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageFilter


DATASET_NAME = "SteelCrack"
SOURCE_SPLIT = "Train"
DEFAULT_SOURCE_ID = "000053"
DEFAULT_SELECTION_COUNT = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(image_path: Path, mask_path: Path) -> dict:
    with Image.open(image_path) as image:
        image_size = image.size
        image_mode = image.mode
    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L")
        mask_size = mask.size
        histogram = mask.histogram()
    if image_size != (512, 512) or mask_size != image_size:
        raise ValueError(
            f"SteelCrack sample {image_path.stem} must be paired 512x512 PNGs; "
            f"image={image_size}, mask={mask_size}"
        )
    invalid_bins = [value for value, count in enumerate(histogram) if count and value not in (0, 255)]
    if invalid_bins:
        raise ValueError(f"SteelCrack mask {mask_path} is not binary 0/255: {invalid_bins[:8]}")
    return {
        "source_id": image_path.stem,
        "image_path": str(image_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "image_sha256": sha256(image_path),
        "mask_sha256": sha256(mask_path),
        "image_mode": image_mode,
        "width": image_size[0],
        "height": image_size[1],
        "crack_pixels": histogram[255],
    }


def _select_records(source_root: Path, count: int, required_id: str) -> list[dict]:
    train_root = source_root / SOURCE_SPLIT
    image_dir = train_root / "images"
    mask_dir = train_root / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected SteelCrack Train/images and Train/masks under {source_root}")

    image_paths = sorted(image_dir.glob("*.png"))
    image_by_id = {path.stem: path for path in image_paths}
    mask_by_id = {path.stem: path for path in mask_dir.glob("*.png")}
    if image_by_id.keys() != mask_by_id.keys():
        raise ValueError("SteelCrack Train image/mask IDs do not match")
    if required_id not in image_by_id:
        raise FileNotFoundError(f"SteelCrack Train source {required_id}.png is missing")

    records = [
        _source_record(image_by_id[source_id], mask_by_id[source_id])
        for source_id in sorted(image_by_id)
    ]
    records = [record for record in records if record["crack_pixels"] > 0]
    if len(records) < count:
        raise ValueError(f"Only {len(records)} non-empty Train masks are available; requested {count}")

    # Deterministic quantile sampling covers thin through dense masks instead of
    # selecting only the visually easiest, largest cracks.
    by_density = sorted(records, key=lambda record: (record["crack_pixels"], record["source_id"]))
    selected: list[dict] = []
    if count == 1:
        indexes = [0]
    else:
        indexes = [round(index * (len(by_density) - 1) / (count - 1)) for index in range(count)]
    for index in indexes:
        record = by_density[index]
        if record["source_id"] not in {item["source_id"] for item in selected}:
            selected.append(record)
    for record in by_density:
        if len(selected) == count:
            break
        if record["source_id"] not in {item["source_id"] for item in selected}:
            selected.append(record)

    required = next(record for record in records if record["source_id"] == required_id)
    selected = [record for record in selected if record["source_id"] != required_id]
    return [required, *selected[: count - 1]]


def _write_rgba(record: dict, output_path: Path, dilation_pixels: int, feather_radius: float) -> dict:
    image_path = Path(record["image_path"])
    mask_path = Path(record["mask_path"])
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    with Image.open(mask_path) as mask_image:
        alpha = mask_image.convert("L")
    if dilation_pixels:
        alpha = alpha.filter(ImageFilter.MaxFilter(dilation_pixels * 2 + 1))
    if feather_radius:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather_radius))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgba = rgb.copy()
    rgba.putalpha(alpha)
    rgba.save(output_path, format="PNG", optimize=True)
    with Image.open(output_path) as written:
        written_alpha = written.getchannel("A")
        alpha_histogram = written_alpha.histogram()
    return {
        "texture_path": str(output_path.resolve()),
        "texture_sha256": sha256(output_path),
        "alpha_nonzero_pixels": sum(alpha_histogram[1:]),
        "alpha_opaque_pixels": alpha_histogram[255],
    }


def prepare_assets(
    source_root: Path,
    output_dir: Path,
    source_id: str = DEFAULT_SOURCE_ID,
    selection_count: int = DEFAULT_SELECTION_COUNT,
    dilation_pixels: int = 2,
    feather_radius: float = 1.0,
) -> dict:
    if selection_count <= 0:
        raise ValueError("selection_count must be positive")
    if dilation_pixels not in (1, 2):
        raise ValueError("dilation_pixels must be 1 or 2")
    if feather_radius < 0:
        raise ValueError("feather_radius must be non-negative")

    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = _select_records(source_root, selection_count, source_id)
    written_records = []
    for record in selected:
        texture_path = output_dir / f"SteelCrack_{record['source_id']}.png"
        generated = _write_rgba(record, texture_path, dilation_pixels, feather_radius)
        written_records.append({**record, **generated})

    primary = written_records[0]
    source_mask_copy = output_dir / f"SteelCrack_{source_id}_source_mask.png"
    with Image.open(primary["mask_path"]) as source_mask:
        source_mask.convert("L").save(source_mask_copy, format="PNG", optimize=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    selection = {
        "dataset": DATASET_NAME,
        "source_split": SOURCE_SPLIT,
        "selection_count": len(written_records),
        "selection_method": "deterministic crack-pixel-density quantiles with required primary source",
        "generated_at_utc": generated_at,
        "dilation_pixels": dilation_pixels,
        "feather_radius": feather_radius,
        "redistribution_boundary": (
            "Internal non-commercial research only; confirm dataset permission before redistribution or commercial use."
        ),
        "sources": written_records,
    }
    selection_path = output_dir / "selected_sources.json"
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    provenance = {
        "dataset": DATASET_NAME,
        "source_split": SOURCE_SPLIT,
        "source_id": source_id,
        "source_image": primary["image_path"],
        "source_mask": primary["mask_path"],
        "source_image_sha256": primary["image_sha256"],
        "source_mask_sha256": primary["mask_sha256"],
        "texture": primary["texture_path"],
        "texture_sha256": primary["texture_sha256"],
        "source_mask_copy": str(source_mask_copy.resolve()),
        "source_mask_copy_sha256": sha256(source_mask_copy),
        "width": primary["width"],
        "height": primary["height"],
        "crack_pixels": primary["crack_pixels"],
        "alpha_nonzero_pixels": primary["alpha_nonzero_pixels"],
        "alpha_opaque_pixels": primary["alpha_opaque_pixels"],
        "dilation_pixels": dilation_pixels,
        "feather_radius": feather_radius,
        "generated_at_utc": generated_at,
        "selection_manifest": str(selection_path.resolve()),
        "redistribution_boundary": selection["redistribution_boundary"],
    }
    provenance_path = output_dir / f"SteelCrack_{source_id}.provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"provenance": provenance, "provenance_path": str(provenance_path), "selection": selection}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/dong/ai/data/external/steelcrack"),
        help="SteelCrack release root containing Train/Validation/Test",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/dong/eun/drone/.runtime/eun-webrtc/crack_assets"),
    )
    parser.add_argument("--source-id", default=DEFAULT_SOURCE_ID)
    parser.add_argument("--selection-count", type=int, default=DEFAULT_SELECTION_COUNT)
    parser.add_argument("--dilation-pixels", type=int, choices=(1, 2), default=2)
    parser.add_argument("--feather-radius", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = prepare_assets(
        args.source_root,
        args.output_dir,
        source_id=args.source_id,
        selection_count=args.selection_count,
        dilation_pixels=args.dilation_pixels,
        feather_radius=args.feather_radius,
    )
    print(json.dumps(result["provenance"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
