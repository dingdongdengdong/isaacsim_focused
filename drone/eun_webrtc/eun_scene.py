#!/usr/bin/env python3
"""Independent EUN WebRTC scene for Isaac Sim 5.1.

This scene uses the local Pegasus extension and a real Iris multirotor, while
remaining independent from the earlier aerial_ws container and PX4.
It is loaded by Isaac Sim's full-streaming application through ``--exec``.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import carb
import omni.kit.app
import omni.usd


STATUS_PATH = Path("/workspace/run/status.json")
CRANE_ASSET_DIR = Path("/workspace/run/crane_assets")
CRACK_ASSET_DIR = Path("/workspace/run/crack_assets")
EVIDENCE_DIR = Path("/workspace/run/evidence/steelcrack-000053")
AERIAL_CRANE_BUILDER = Path("/workspace/aerial/scripts/build_port_cranes.py")
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from steelcrack_usd import (  # noqa: E402
    INSPECTION_CAMERA,
    PRIMARY_DECAL_PRIM,
    SEMANTIC_LABEL,
    define_camera,
    define_decal,
    inspection_camera_poses,
    load_json,
)


def set_color(prim, color: tuple[float, float, float]) -> None:
    from pxr import UsdGeom

    imageable = UsdGeom.Gprim(prim)
    imageable.CreateDisplayColorAttr([color])


def add_cube(stage, path: str, position, scale, color) -> None:
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*position))
    cube.AddScaleOp().Set(Gf.Vec3f(*scale))
    set_color(cube.GetPrim(), color)


def add_cylinder(stage, path: str, position, radius, height, color) -> None:
    from pxr import Gf, UsdGeom

    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateAxisAttr("Z")
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    cylinder.AddTranslateOp().Set(Gf.Vec3d(*position))
    set_color(cylinder.GetPrim(), color)


def add_drone_camera(stage, path: str) -> None:
    """Create a forward-facing USD camera rigidly attached to the Iris body."""
    from pxr import Gf, UsdGeom

    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500.0))
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(0.30, 0.0, 0.0))
    # USD cameras look down local -Z. Rotate -90 degrees about Y so -Z is body +X.
    xform.AddRotateYOp().Set(-90.0)


async def capture_viewport(viewport, path: Path, frames: int = 12) -> str:
    from omni.kit.viewport.utility import capture_viewport_to_file, next_viewport_frame_async

    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(frames):
        try:
            await next_viewport_frame_async(viewport)
        except TypeError:
            await next_viewport_frame_async()
    # ``Capture.wait_for_result`` is a blocking call.  Calling it from this
    # ``--exec`` coroutine starves Kit's update loop, so keep the capture
    # object alive and advance viewport frames until its file is complete.
    capture = capture_viewport_to_file(viewport, file_path=str(path))
    previous_size = -1
    stable_frames = 0
    for _ in range(180):
        try:
            await next_viewport_frame_async(viewport)
        except TypeError:
            await next_viewport_frame_async()
        size = path.stat().st_size if path.is_file() else -1
        if size > 0 and size == previous_size:
            stable_frames += 1
            if stable_frames >= 2:
                break
        else:
            stable_frames = 0
        previous_size = size
    else:
        raise TimeoutError(f"viewport capture did not complete: {path}")
    del capture
    return str(path)


async def build_scene() -> None:
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Gf, UsdGeom, UsdLux
    from isaacsim.core.utils.extensions import enable_extension

    app = omni.kit.app.get_app()

    enable_extension("isaacsim.ros2.bridge")
    for _ in range(20):
        await app.next_update_async()

    extension_id = ""
    for _ in range(300):
        extension_id = app.get_extension_manager().get_enabled_extension_id(
            "pegasus.simulator"
        )
        if extension_id:
            break
        await app.next_update_async()
    if not extension_id:
        raise RuntimeError("pegasus.simulator is not enabled")

    from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
    from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
    from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS, WORLD_SETTINGS
    from ros2_flight_controller import (
        CMD_VEL_TOPIC,
        ROTOR_TOPICS,
        FixedRotorROS2Backend,
        Ros2VelocityController,
    )
    from flight_control_core import VelocityCommand

    pegasus = PegasusInterface()
    pegasus.set_world_settings(**WORLD_SETTINGS["px4"])
    await pegasus.load_environment_async(
        SIMULATION_ENVIRONMENTS["Flat Plane"], force_clear=True
    )
    stage = omni.usd.get_context().get_stage()

    # Generate the two crane USDs inside this EUN runtime; source assets remain untouched.
    spec = importlib.util.spec_from_file_location(
        "eun_build_port_cranes", AERIAL_CRANE_BUILDER
    )
    cranes = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load crane builder: {AERIAL_CRANE_BUILDER}")
    spec.loader.exec_module(cranes)
    cranes.OUTPUT_DIR = str(CRANE_ASSET_DIR)
    CRANE_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    cranes.build_container_crane()
    cranes.build_transfer_crane()

    crane_specs = (
        ("container_crane.usd", "/World/ContainerCrane", "/ContainerCrane", (-30.0, -26.0, 0.0)),
        ("transfer_crane.usd", "/World/TransferCrane", "/TransferCrane", (20.0, -10.0, 0.0)),
    )
    crane_children = {}
    for filename, prim_path, asset_prim, position in crane_specs:
        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(str(CRANE_ASSET_DIR / filename), asset_prim)
        xform = UsdGeom.Xformable(prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        xform.AddRotateXOp().Set(90.0)

    for _ in range(40):
        await app.next_update_async()
    for _, prim_path, _, _ in crane_specs:
        prim = stage.GetPrimAtPath(prim_path)
        child_count = len(list(prim.GetChildren())) if prim.IsValid() else 0
        if child_count == 0:
            raise RuntimeError(f"crane reference did not compose: {prim_path}")
        crane_children[prim_path] = child_count

    provenance_path = CRACK_ASSET_DIR / "SteelCrack_000053.provenance.json"
    provenance = load_json(provenance_path)
    texture_path = CRACK_ASSET_DIR / "SteelCrack_000053.png"
    if provenance["source_split"] != "Train" or not texture_path.is_file():
        raise RuntimeError("SteelCrack Train RGBA asset/provenance is unavailable")
    define_decal(
        stage,
        str(texture_path),
        position=(0.0, 20.0, -4.005),
        size=(2.0, 1.5),
    )

    key = UsdLux.DistantLight.Define(stage, "/World/MoonLight")
    key.CreateIntensityAttr(4000.0)
    key.CreateColorAttr(Gf.Vec3f(0.72, 0.82, 1.0))
    key.AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 25.0, -20.0))

    dome = UsdLux.DomeLight.Define(stage, "/World/Ambient")
    dome.CreateIntensityAttr(1800.0)
    dome.CreateColorAttr(Gf.Vec3f(0.45, 0.52, 0.72))

    inspection_light = UsdLux.SphereLight.Define(stage, "/World/InspectionLight")
    inspection_light.CreateIntensityAttr(25000.0)
    inspection_light.CreateRadiusAttr(1.5)
    inspection_light.CreateColorAttr(Gf.Vec3f(1.0, 0.82, 0.64))
    inspection_light.AddTranslateOp().Set(Gf.Vec3d(26.0, 8.0, 30.0))

    vehicle_config = MultirotorConfig()
    print("EUN_CONTROL_SETUP=creating_ros2_backend", flush=True)
    ros2_backend = FixedRotorROS2Backend(
        vehicle_id=0,
        config={
            "namespace": "/drone",
            "pub_graphical_sensors": False,
            "pub_sensors": True,
            "pub_state": True,
            "pub_tf": True,
            "sub_control": True,
        },
    )
    print("EUN_CONTROL_SETUP=creating_velocity_controller", flush=True)
    velocity_controller = Ros2VelocityController()
    velocity_controller.attach_ros_node(ros2_backend.node)
    vehicle_config.backends = [ros2_backend, velocity_controller]
    print("EUN_CONTROL_SETUP=creating_multirotor", flush=True)
    vehicle = Multirotor(
        "/World/eun_iris",
        ROBOTS["Iris"],
        0,
        [8.0, 8.0, 18.0],
        [0.0, 0.0, 0.0, 1.0],
        config=vehicle_config,
    )
    physics_probe = {"steps": 0, "last_dt": 0.0}

    def record_physics_step(dt: float) -> None:
        physics_probe["steps"] += 1
        physics_probe["last_dt"] = float(dt)

    pegasus.world.add_physics_callback("/World/eun_runtime_probe", record_physics_step)
    print("EUN_CONTROL_SETUP=creating_drone_camera", flush=True)
    drone_camera_path = "/World/eun_iris/body/EunFpvCamera"
    add_drone_camera(stage, drone_camera_path)
    print("EUN_CONTROL_SETUP=resetting_world", flush=True)
    await pegasus.world.reset_async()
    print("EUN_CONTROL_SETUP=world_reset", flush=True)

    # Preserve Pegasus' overview camera and add persistent 4.2 m inspection
    # cameras. The front inspection camera is the default WebRTC/world view.
    pegasus.set_viewport_camera(
        camera_position=[48.0, 32.0, 31.0],
        camera_target=[19.0, -8.0, 19.0],
    )
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("streaming app has no active viewport")
    viewport.resolution = (1280, 720)
    overview_camera_path = str(viewport.camera_path)
    inspection_target = (20.0, -5.995, 20.0)
    inspection_poses = inspection_camera_poses(target=inspection_target, distance=4.2)
    inspection_camera_paths = {}
    for name, eye in inspection_poses.items():
        path = INSPECTION_CAMERA if name == "front" else f"/World/Cameras/CrackInspectionCamera_{name}"
        define_camera(stage, path, eye, inspection_target)
        inspection_camera_paths[name] = path
    world_camera_path = INSPECTION_CAMERA
    viewport.camera_path = world_camera_path
    if not stage.GetPrimAtPath(drone_camera_path).IsValid():
        raise RuntimeError(f"EUN drone camera is missing: {drone_camera_path}")
    velocity_controller.attach_keyboard(
        viewport,
        world_camera_path=world_camera_path,
        drone_camera_path=drone_camera_path,
    )
    print("EUN_CONTROL_SETUP=playing_world", flush=True)
    await pegasus.world.play_async()
    print("EUN_CONTROL_SETUP=world_playing", flush=True)
    for _ in range(30):
        await app.next_update_async()

    # Keep the streaming service readiness independent from image readback.
    # Headless viewport capture can wait indefinitely when no WebRTC client is
    # consuming frames; deterministic RGB proof is produced by eun_sdg.py.
    evidence = {}
    capture_errors = {"viewport": "deferred_to_headless_basic_writer"}
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    viewport.camera_path = world_camera_path
    evidence["binary_source_mask"] = str(EVIDENCE_DIR / "binary_mask.png")
    source_mask_in_container = provenance["source_mask_copy"].replace(
        "/home/dong/eun/drone/.runtime/eun-webrtc", "/workspace/run"
    )
    shutil.copyfile(source_mask_in_container, evidence["binary_source_mask"])
    stage.Export(str(EVIDENCE_DIR / "scene.usda"))
    evidence["scene_usda"] = str(EVIDENCE_DIR / "scene.usda")
    for artifact in EVIDENCE_DIR.rglob("*"):
        artifact.chmod(0o777 if artifact.is_dir() else 0o666)
    EVIDENCE_DIR.chmod(0o777)

    ros2_self_test = {
        "enabled": os.environ.get("EUN_ROS2_SELF_TEST", "0") == "1",
        "initial_position": None,
        "final_position": None,
        "delta": None,
    }
    if ros2_self_test["enabled"]:
        initial_position = [float(value) for value in vehicle.state.position]
        ros2_self_test["initial_position"] = initial_position
        for _ in range(60):
            velocity_controller.publish_command(VelocityCommand(forward=0.6))
            await app.next_update_async()
        for _ in range(30):
            velocity_controller.publish_command(VelocityCommand())
            await app.next_update_async()
        final_position = [float(value) for value in vehicle.state.position]
        ros2_self_test["final_position"] = final_position
        ros2_self_test["delta"] = [
            final_position[index] - initial_position[index] for index in range(3)
        ]

    status = {
        "ready": True,
        "runtime": "Isaac Sim 5.1 full streaming",
        "scene": "EUN Crane Crack Inspection",
        "pegasus_extension_enabled": True,
        "pegasus_extension_id": extension_id,
        "ros2_bridge_enabled": True,
        "ros2_control": True,
        "ros2_cmd_vel_topic": CMD_VEL_TOPIC,
        "ros2_rotor_topics": list(ROTOR_TOPICS),
        "ros2_state_topics": [
            "/drone0/state/pose",
            "/drone0/state/twist",
            "/drone0/state/twist_inertial",
        ],
        "ros2_cmd_vel_receive_count_at_ready": velocity_controller.cmd_vel_receive_count,
        "ros2_rotor_receive_counts_at_ready": list(ros2_backend.rotor_receive_counts),
        "ros2_last_rotor_reference_at_ready": velocity_controller.last_rotor_reference,
        "ros2_self_test": ros2_self_test,
        "control_path": "WASD -> ROS 2 Twist -> velocity controller -> ROS 2 rotor refs -> Pegasus dynamics",
        "keyboard_controls": {
            "W/S": "forward/backward",
            "A/D": "left/right",
            "R/F": "up/down",
            "Q/E": "yaw left/right",
            "C": "world/drone camera toggle",
        },
        "vehicle_loaded": stage.GetPrimAtPath("/World/eun_iris").IsValid(),
        "vehicle_prim": "/World/eun_iris",
        "vehicle_type": "Pegasus Iris Multirotor",
        "vehicle_object_created": vehicle is not None,
        "physics_step_count_at_ready": physics_probe["steps"],
        "physics_last_dt_at_ready": physics_probe["last_dt"],
        "vehicle_position_at_ready": [float(value) for value in vehicle.state.position],
        "cranes_loaded": True,
        "crane_children": crane_children,
        "crack_loaded": stage.GetPrimAtPath(PRIMARY_DECAL_PRIM).IsValid(),
        "crack_mode": "rgba_decal",
        "source_dataset": "SteelCrack",
        "source_split": provenance["source_split"],
        "source_id": provenance["source_id"],
        "texture_sha256": provenance["texture_sha256"],
        "decal_prim": PRIMARY_DECAL_PRIM,
        "inspection_camera": INSPECTION_CAMERA,
        "inspection_distance_m": 4.2,
        "semantic_label": SEMANTIC_LABEL,
        "crack_provenance": str(provenance_path),
        "crack_target": "/World/TransferCrane main girder camera-facing steel surface",
        "independent_from": ["earlier aerial_ws container", "PX4"],
        "signal_port": 49101,
        "media_port": 47999,
        "public_endpoint": "100.96.41.100",
        "camera_path": world_camera_path,
        "world_camera_path": world_camera_path,
        "overview_camera_path": overview_camera_path,
        "drone_camera_path": drone_camera_path,
        "drone_camera_loaded": stage.GetPrimAtPath(drone_camera_path).IsValid(),
        "camera_toggle_key": "C",
        "viewport_resolution": [1280, 720],
        "camera_method": "persistent crack inspection USD camera plus Iris body camera toggle",
        "evidence": evidence,
        "capture_errors": capture_errors,
        "headless_rgb_probe": "deferred_to_headless_basic_writer",
        "visual_validation": "saved PNGs require direct review; runtime structure alone is not visual proof",
    }
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print("EUN_WEBRTC_READY=" + json.dumps(status, sort_keys=True), flush=True)
    carb.log_info("EUN independent WebRTC scene is ready")


asyncio.ensure_future(build_scene())
