#!/usr/bin/env python3
"""Isaac Sim 5.1 BasicWriter job for SteelCrack crane-decal SDG."""
from __future__ import annotations

import asyncio
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import carb
import omni.kit.app
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, UsdGeom, UsdLux


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from steelcrack_usd import (  # noqa: E402
    PRIMARY_DECAL_PRIM,
    SEMANTIC_LABEL,
    define_camera,
    define_decal,
    set_camera_pose,
    set_texture,
    transfer_local_to_world,
    world_decal_corners,
)


RUN_ROOT = Path("/workspace/run")
CRACK_ASSET_DIR = RUN_ROOT / "crack_assets"
CRANE_ASSET_DIR = RUN_ROOT / "crane_assets"
OUTPUT_ROOT = RUN_ROOT / "sdg" / os.environ.get("EUN_SDG_NAME", "steelcrack-smoke-100")
FRAME_COUNT = int(os.environ.get("EUN_SDG_FRAMES", "100"))
SEED = int(os.environ.get("EUN_SDG_SEED", "20260815"))
RESOLUTION = (512, 512)
CAMERA_PATH = "/World/Cameras/SteelCrackSDGCamera"


def _writer_path(prefix: str, frame_id: int, suffix: str) -> str:
    return f"{prefix}_{frame_id:04d}.{suffix}"


def _write_status(status: dict) -> None:
    path = OUTPUT_ROOT / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_host_writable() -> None:
    for path in sorted(OUTPUT_ROOT.rglob("*")):
        path.chmod(0o777 if path.is_dir() else 0o666)
    OUTPUT_ROOT.chmod(0o777)


async def generate() -> None:
    app = omni.kit.app.get_app()
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty SDG output: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    raw_dir = OUTPUT_ROOT / "basic_writer"
    raw_dir.mkdir()
    projected_mask_dir = OUTPUT_ROOT / "masks"
    projected_mask_dir.mkdir()

    selection = json.loads((CRACK_ASSET_DIR / "selected_sources.json").read_text())
    if selection["dataset"] != "SteelCrack" or selection["source_split"] != "Train":
        raise RuntimeError("SDG source selection must be SteelCrack Train")
    sources = selection["sources"]
    if len(sources) != 64:
        raise RuntimeError(f"Expected 64 selected SteelCrack Train sources, found {len(sources)}")

    omni.usd.get_context().new_stage()
    rep.orchestrator.set_capture_on_play(False)
    rep.set_global_seed(SEED)
    stage = omni.usd.get_context().get_stage()

    crane = stage.DefinePrim("/World/TransferCrane", "Xform")
    crane.GetReferences().AddReference(
        str(CRANE_ASSET_DIR / "transfer_crane.usd"), "/TransferCrane"
    )
    crane_xform = UsdGeom.Xformable(crane)
    crane_xform.AddTranslateOp().Set(Gf.Vec3d(20.0, -10.0, 0.0))
    crane_xform.AddRotateXOp().Set(90.0)
    for _ in range(30):
        await app.next_update_async()
    if not list(crane.GetChildren()):
        raise RuntimeError("TransferCrane reference did not compose in SDG stage")

    primary_texture = CRACK_ASSET_DIR / f"SteelCrack_{sources[0]['source_id']}.png"
    decal = define_decal(stage, str(primary_texture))
    mesh = decal["mesh"]

    dome = UsdLux.DomeLight.Define(stage, "/World/SDG/Lights/Dome")
    dome.CreateIntensityAttr(600.0)
    dome.CreateColorAttr(Gf.Vec3f(0.75, 0.82, 1.0))
    key = UsdLux.SphereLight.Define(stage, "/World/SDG/Lights/Inspection")
    key.CreateRadiusAttr(1.0)
    key.CreateIntensityAttr(18000.0)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.9, 0.75))
    key_translate = key.AddTranslateOp()
    key_translate.Set(Gf.Vec3d(20.0, -1.0, 22.0))

    initial_target = transfer_local_to_world((0.0, 20.0, -4.005))
    _, camera_transform = define_camera(
        stage,
        CAMERA_PATH,
        (20.0, -1.5, 20.0),
        initial_target,
        focal_length=35.0,
        aperture=20.955,
    )

    render_product = rep.create.render_product(CAMERA_PATH, RESOLUTION)
    # Initialize Replicator before attaching BasicWriter. In Kit-hosted --exec
    # workflows the first orchestrator step initializes the graph and emits no
    # writer frame.
    await rep.orchestrator.step_async(rt_subframes=2)
    for _ in range(4):
        await app.next_update_async()

    writer = rep.writers.get("BasicWriter")
    writer.initialize(
        output_dir=str(raw_dir),
        rgb=True,
        semantic_segmentation=True,
        colorize_semantic_segmentation=False,
        camera_params=True,
        frame_padding=4,
    )
    writer.attach(render_product)
    for _ in range(4):
        await app.next_update_async()

    manifest_path = OUTPUT_ROOT / "manifest.jsonl"
    positive_index = 0
    started_at = datetime.now(timezone.utc).isoformat()
    _write_status(
        {
            "state": "running",
            "started_at_utc": started_at,
            "frames_requested": FRAME_COUNT,
            "frames_submitted": 0,
            "seed": SEED,
        }
    )
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for frame_id in range(FRAME_COUNT):
            frame_seed = SEED + frame_id
            frame_rng = random.Random(frame_seed)
            clean = frame_id % 5 == 0
            source = None if clean else sources[positive_index % len(sources)]
            if not clean:
                positive_index += 1

            width = frame_rng.uniform(0.8, 2.4)
            height = frame_rng.uniform(0.45, 1.3)
            rotation = frame_rng.uniform(-18.0, 18.0)
            local_position = (
                frame_rng.uniform(-8.5, 8.5),
                frame_rng.uniform(19.0 + height / 2.0, 21.0 - height / 2.0),
                -4.005,
            )
            mesh.GetPrim().SetActive(True)
            decal["translate_op"].Set(Gf.Vec3d(*local_position))
            decal["rotate_op"].Set(rotation)
            decal["scale_op"].Set(Gf.Vec3f(width, height, 1.0))
            if clean:
                mesh.GetPrim().SetActive(False)
                source_id = None
                source_texture = None
            else:
                source_id = source["source_id"]
                source_texture = CRACK_ASSET_DIR / f"SteelCrack_{source_id}.png"
                set_texture(decal["texture"], str(source_texture))

            target = transfer_local_to_world(local_position)
            angle = frame_rng.uniform(-30.0, 30.0)
            distance = frame_rng.uniform(3.0, 5.0)
            vertical_offset = frame_rng.uniform(-0.35, 0.35)
            angle_radians = math.radians(angle)
            eye = (
                target[0] + math.sin(angle_radians) * distance,
                target[1] + math.cos(angle_radians) * distance,
                target[2] + vertical_offset,
            )
            set_camera_pose(camera_transform, eye, target)

            dome_intensity = frame_rng.uniform(250.0, 900.0)
            key_intensity = frame_rng.uniform(8000.0, 30000.0)
            color_temperature_mix = frame_rng.uniform(0.0, 1.0)
            dome.GetIntensityAttr().Set(dome_intensity)
            key.GetIntensityAttr().Set(key_intensity)
            key.GetColorAttr().Set(
                Gf.Vec3f(
                    1.0,
                    0.78 + 0.18 * color_temperature_mix,
                    0.62 + 0.30 * color_temperature_mix,
                )
            )
            key_translate.Set(
                Gf.Vec3d(
                    target[0] + frame_rng.uniform(-2.0, 2.0),
                    target[1] + frame_rng.uniform(2.0, 5.0),
                    target[2] + frame_rng.uniform(-1.0, 2.0),
                )
            )

            for _ in range(2):
                await app.next_update_async()
            await rep.orchestrator.step_async(rt_subframes=2)

            manifest_record = {
                "frame_id": frame_id,
                "seed": frame_seed,
                "clean_hard_negative": clean,
                "source_dataset": "SteelCrack",
                "source_split": "Train",
                "source_id": source_id,
                "source_texture": str(source_texture) if source_texture else None,
                "decal_prim": PRIMARY_DECAL_PRIM,
                "decal_pose_local": {
                    "translation": list(local_position),
                    "rotation_degrees": rotation,
                },
                "decal_scale_m": [width, height],
                "decal_world_corners": [
                    list(point) for point in world_decal_corners(local_position, (width, height), rotation)
                ],
                "camera_prim": CAMERA_PATH,
                "camera_pose": {"eye": list(eye), "target": list(target), "distance_m": distance},
                "camera_intrinsics": {
                    "focal_length_mm": 35.0,
                    "horizontal_aperture_mm": 20.955,
                    "vertical_aperture_mm": 20.955,
                    "resolution": list(RESOLUTION),
                },
                "lighting": {
                    "dome_intensity": dome_intensity,
                    "inspection_intensity": key_intensity,
                    "warmth_mix": color_temperature_mix,
                },
                "rgb_path": f"basic_writer/{_writer_path('rgb', frame_id, 'png')}",
                "semantic_mask_path": f"basic_writer/{_writer_path('semantic_segmentation', frame_id, 'png')}",
                "semantic_labels_path": f"basic_writer/{_writer_path('semantic_segmentation_labels', frame_id, 'json')}",
                "camera_params_path": f"basic_writer/{_writer_path('camera_params', frame_id, 'json')}",
                "mask_path": f"masks/frame_{frame_id:05d}.png",
                "semantic_label": SEMANTIC_LABEL,
            }
            manifest.write(json.dumps(manifest_record, sort_keys=True) + "\n")
            manifest.flush()
            if (frame_id + 1) % 10 == 0 or frame_id + 1 == FRAME_COUNT:
                _write_status(
                    {
                        "state": "running",
                        "started_at_utc": started_at,
                        "frames_requested": FRAME_COUNT,
                        "frames_submitted": frame_id + 1,
                        "seed": SEED,
                    }
                )

    await rep.orchestrator.wait_until_complete_async()
    for _ in range(10):
        await app.next_update_async()
    written_frames = int(getattr(writer, "_frame_id", -1))
    expected_first_rgb = raw_dir / _writer_path("rgb", 0, "png")
    if written_frames != FRAME_COUNT or not expected_first_rgb.is_file():
        raise RuntimeError(
            f"BasicWriter did not persist the requested frames: "
            f"writer_frame_id={written_frames}, expected={FRAME_COUNT}, "
            f"first_rgb_exists={expected_first_rgb.is_file()}"
        )
    writer.detach()
    render_product.destroy()
    stage.Export(str(OUTPUT_ROOT / "scene.usda"))
    completed_at = datetime.now(timezone.utc).isoformat()
    _write_status(
        {
            "state": "generated",
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "frames_requested": FRAME_COUNT,
            "frames_submitted": FRAME_COUNT,
            "seed": SEED,
            "manifest": str(manifest_path),
            "writer": "Isaac Sim 5.1 BasicWriter",
            "writer_frames": written_frames,
        }
    )
    _make_host_writable()
    print(f"EUN_STEELCRACK_SDG_COMPLETE={OUTPUT_ROOT}", flush=True)
    app.post_quit(0)


async def guarded_generate() -> None:
    try:
        await generate()
    except Exception as exc:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        _write_status({"state": "failed", "error": repr(exc), "seed": SEED})
        _make_host_writable()
        carb.log_error(f"SteelCrack SDG failed: {exc!r}")
        omni.kit.app.get_app().post_quit(1)
        raise


asyncio.ensure_future(guarded_generate())
