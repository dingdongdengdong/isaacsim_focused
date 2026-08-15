#!/usr/bin/env python3
"""Render deterministic overview and crack-inspection evidence from scene.usda."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.usd
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from steelcrack_usd import define_camera, inspection_camera_poses, set_camera_pose  # noqa: E402


EVIDENCE_DIR = Path("/workspace/run/evidence/steelcrack-000053")
SCENE_PATH = EVIDENCE_DIR / "scene.usda"
CAMERA_PATH = "/World/Cameras/SteelCrackEvidenceCamera"
RESOLUTION = (1280, 720)


async def capture() -> None:
    app = omni.kit.app.get_app()
    if not SCENE_PATH.is_file():
        raise FileNotFoundError(SCENE_PATH)
    opened = await omni.usd.get_context().open_stage_async(str(SCENE_PATH))
    if isinstance(opened, tuple) and opened and not opened[0]:
        raise RuntimeError(f"failed to open stage: {opened}")
    for _ in range(30):
        await app.next_update_async()
    stage = omni.usd.get_context().get_stage()

    target = (20.0, -5.995, 20.0)
    poses = {
        "overview": ((180.0, 160.0, 105.0), (0.0, -10.0, 25.0)),
        **{name: (eye, target) for name, eye in inspection_camera_poses(target, 4.2).items()},
    }
    camera, camera_transform = define_camera(stage, CAMERA_PATH, *poses["front"])
    render_product = rep.create.render_product(CAMERA_PATH, RESOLUTION)
    rep.orchestrator.set_capture_on_play(False)
    await rep.orchestrator.step_async(rt_subframes=4)
    for _ in range(4):
        await app.next_update_async()

    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(render_product)
    outputs = {}
    for name, (eye, look_target) in poses.items():
        camera.GetFocalLengthAttr().Set(20.0 if name == "overview" else 35.0)
        set_camera_pose(camera_transform, eye, look_target)
        for _ in range(4):
            await app.next_update_async()
        await rep.orchestrator.step_async(rt_subframes=4)
        pixels = np.asarray(rgb.get_data())
        if pixels.shape[:2] != (RESOLUTION[1], RESOLUTION[0]):
            raise RuntimeError(f"unexpected RGB shape for {name}: {pixels.shape}")
        path = EVIDENCE_DIR / ("overview.png" if name == "overview" else f"inspection_{name}.png")
        Image.fromarray(pixels).save(path)
        path.chmod(0o666)
        outputs[name] = str(path)

    rgb.detach(render_product)
    render_product.destroy()
    status = {
        "state": "captured",
        "scene": str(SCENE_PATH),
        "resolution": list(RESOLUTION),
        "outputs": outputs,
    }
    status_path = EVIDENCE_DIR / "capture_status.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    status_path.chmod(0o666)
    EVIDENCE_DIR.chmod(0o777)
    print("EUN_SCENE_EVIDENCE=" + json.dumps(status, sort_keys=True), flush=True)
    app.post_quit(0)


async def guarded_capture() -> None:
    try:
        await capture()
    except Exception as exc:
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        (EVIDENCE_DIR / "capture_status.json").write_text(
            json.dumps({"state": "failed", "error": repr(exc)}, indent=2) + "\n"
        )
        omni.kit.app.get_app().post_quit(1)
        raise


asyncio.ensure_future(guarded_capture())
