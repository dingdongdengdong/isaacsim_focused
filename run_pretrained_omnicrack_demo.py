#!/usr/bin/env python3
"""Create a reviewed visual demo with the official OmniCrack30k model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


DEFAULT_MODEL_ROOT = Path(
    "/home/dong/ai/external/omnicrack30k/src"
)
DEFAULT_DATASET_ROOT = Path("/home/dong/ai/data/steelcrack")
DEFAULT_OUTPUT_ROOT = Path("runs/omnicrack30k-pretrained-demo")
DEFAULT_SAMPLE_IDS = tuple(f"{index:06d}" for index in range(1, 13))


@dataclass(frozen=True)
class BinaryMetrics:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    iou: float
    dice: float
    precision: float
    recall: float


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _metrics(prediction: np.ndarray, target: np.ndarray) -> BinaryMetrics:
    true_positive = int(np.logical_and(prediction, target).sum())
    false_positive = int(np.logical_and(prediction, ~target).sum())
    false_negative = int(np.logical_and(~prediction, target).sum())
    true_negative = int(np.logical_and(~prediction, ~target).sum())
    return BinaryMetrics(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        iou=_ratio(true_positive, true_positive + false_positive + false_negative),
        dice=_ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative),
        precision=_ratio(true_positive, true_positive + false_positive),
        recall=_ratio(true_positive, true_positive + false_negative),
    )


def _aggregate(items: list[BinaryMetrics]) -> BinaryMetrics:
    totals = {
        name: sum(getattr(item, name) for item in items)
        for name in ("true_positive", "false_positive", "false_negative", "true_negative")
    }
    return _metrics_from_totals(**totals)


def _metrics_from_totals(
    true_positive: int,
    false_positive: int,
    false_negative: int,
    true_negative: int,
) -> BinaryMetrics:
    return BinaryMetrics(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        iou=_ratio(true_positive, true_positive + false_positive + false_negative),
        dice=_ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative),
        precision=_ratio(true_positive, true_positive + false_positive),
        recall=_ratio(true_positive, true_positive + false_negative),
    )


def _label(image: Image.Image, text: str) -> Image.Image:
    rendered = image.convert("RGB")
    draw = ImageDraw.Draw(rendered)
    draw.rectangle((0, 0, rendered.width, 28), fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255))
    return rendered


def _probability_heatmap(probability: np.ndarray) -> Image.Image:
    heatmap = cv2.applyColorMap(
        np.uint8(np.clip(probability, 0.0, 1.0) * 255), cv2.COLORMAP_TURBO
    )
    return Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))


def _overlay(
    image: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    centerline: np.ndarray,
) -> Image.Image:
    rendered = image.astype(np.float32).copy()
    true_positive = prediction & target
    false_positive = prediction & ~target
    false_negative = ~prediction & target
    rendered[true_positive] = 0.45 * rendered[true_positive] + 0.55 * np.array([0, 220, 0])
    rendered[false_positive] = 0.35 * rendered[false_positive] + 0.65 * np.array([255, 210, 0])
    rendered[false_negative] = 0.35 * rendered[false_negative] + 0.65 * np.array([255, 0, 0])
    rendered[centerline] = np.array([0, 200, 255])
    return Image.fromarray(np.uint8(np.clip(rendered, 0, 255)))


def _panel(
    image: np.ndarray,
    target: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    centerline: np.ndarray,
    metrics: BinaryMetrics,
) -> Image.Image:
    height, width = target.shape
    tiles = [
        _label(Image.fromarray(image), "Input"),
        _label(Image.fromarray(np.uint8(target) * 255), "Ground truth"),
        _label(_probability_heatmap(probability), "Crack probability"),
        _label(Image.fromarray(np.uint8(prediction) * 255), "Prediction"),
        _label(Image.fromarray(np.uint8(centerline) * 255), "Centerline"),
        _label(
            _overlay(image, prediction, target, centerline),
            f"TP green | FP yellow | FN red | Dice {metrics.dice:.3f} | Recall {metrics.recall:.3f}",
        ),
    ]
    canvas = Image.new("RGB", (width * len(tiles), height), "white")
    for index, tile in enumerate(tiles):
        canvas.paste(tile, (index * width, 0))
    return canvas


def _contact_sheet(panels: list[Image.Image], output: Path, width: int = 1800) -> None:
    resized: list[Image.Image] = []
    for panel in panels:
        height = max(1, round(panel.height * width / panel.width))
        resized.append(panel.resize((width, height), Image.Resampling.LANCZOS))
    sheet = Image.new("RGB", (width, sum(panel.height for panel in resized)), "white")
    y = 0
    for panel in resized:
        sheet.paste(panel, (0, y))
        y += panel.height
    sheet.save(output, quality=94)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-ids", nargs="+", default=list(DEFAULT_SAMPLE_IDS))
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 4])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sys.path.insert(0, str(args.model_root))
    from omnicrack30k.inference import OmniCrack30kModel

    image_root = args.dataset_root / "images" / "test"
    mask_root = args.dataset_root / "masks" / "test"
    for sample_id in args.sample_ids:
        for path in (image_root / f"{sample_id}.png", mask_root / f"{sample_id}.png"):
            if not path.exists():
                raise FileNotFoundError(path)

    args.output_root.mkdir(parents=True, exist_ok=True)
    model = OmniCrack30kModel(folds=tuple(args.folds), allow_tqdm=False)
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    records: list[dict[str, object]] = []
    metric_items: list[BinaryMetrics] = []
    panels: list[Image.Image] = []
    for sample_id in args.sample_ids:
        image_path = image_root / f"{sample_id}.png"
        mask_path = mask_root / f"{sample_id}.png"
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read {image_path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        target = np.asarray(Image.open(mask_path).convert("L")) == 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        softmax_visual, mask_visual, centerline_visual = model.predict_np(image, rgb=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        # The upstream visual outputs use zero/black for crack. Normalize all
        # saved outputs to True/white = crack to prevent the prior polarity bug.
        probability = 1.0 - np.asarray(softmax_visual, dtype=np.float32)
        prediction = np.asarray(mask_visual) == 0
        centerline = np.asarray(centerline_visual) == 0
        metrics = _metrics(prediction, target)
        metric_items.append(metrics)

        sample_root = args.output_root / "samples" / sample_id
        sample_root.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(sample_root / "input.png")
        Image.fromarray(np.uint8(target) * 255).save(sample_root / "ground_truth.png")
        Image.fromarray(np.uint8(prediction) * 255).save(sample_root / "prediction.png")
        Image.fromarray(np.uint8(centerline) * 255).save(sample_root / "centerline.png")
        _probability_heatmap(probability).save(sample_root / "probability.png")
        overlay = _overlay(image, prediction, target, centerline)
        overlay.save(sample_root / "overlay.png")
        panel = _panel(image, target, probability, prediction, centerline, metrics)
        panel.save(sample_root / "panel.jpg", quality=94)
        panels.append(panel)
        records.append(
            {
                "sample_id": sample_id,
                "image": str(image_path),
                "mask": str(mask_path),
                "seconds": elapsed,
                "metrics": asdict(metrics),
                "target_crack_fraction": float(target.mean()),
                "predicted_crack_fraction": float(prediction.mean()),
                "output": str(sample_root.resolve()),
            }
        )
        print(
            f"{sample_id}: dice={metrics.dice:.4f} recall={metrics.recall:.4f} "
            f"precision={metrics.precision:.4f} seconds={elapsed:.3f}",
            flush=True,
        )

    _contact_sheet(panels, args.output_root / "contact-sheet.jpg")
    aggregate = _aggregate(metric_items)
    known_sample_gate = next(item for item in records if item["sample_id"] == "000001")
    known_metrics = known_sample_gate["metrics"]
    gate_applicable = args.folds == [0]
    gate_passed = bool(
        known_metrics["dice"] >= 0.85 and known_metrics["recall"] >= 0.90
    )
    gate = {
        "sample_id": "000001",
        "reference": "single-fold-0 polarity and preprocessing reproduction",
        "applicable": gate_applicable,
        "dice_min": 0.85,
        "recall_min": 0.90,
        "passed": gate_passed if gate_applicable else None,
    }
    report = {
        "model": "official OmniCrack30k nnU-Net 2D",
        "source": "https://github.com/ben-z-original/omnicrack30k",
        "folds": args.folds,
        "device": device,
        "dataset": "SteelCrack held-out test showcase",
        "samples": len(records),
        "aggregate_metrics": asdict(aggregate),
        "known_sample_gate": gate,
        "records": records,
        "artifacts": {
            "root": str(args.output_root.resolve()),
            "contact_sheet": str((args.output_root / "contact-sheet.jpg").resolve()),
        },
        "scope": "Visual pretrained-model showcase; not a field, safety, or repair-authorization claim.",
    }
    (args.output_root / "results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in report if key != "records"}, indent=2))
    if gate_applicable and not gate_passed:
        raise SystemExit("Known-sample validation gate failed")


if __name__ == "__main__":
    main()
