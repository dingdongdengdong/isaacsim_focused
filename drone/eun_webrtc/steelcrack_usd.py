"""USD helpers shared by the EUN streaming scene and SteelCrack SDG job."""
from __future__ import annotations

import json
import math
from pathlib import Path


DECAL_ROOT = "/World/TransferCrane/CrackDecals"
PRIMARY_DECAL_PRIM = f"{DECAL_ROOT}/SteelCrack_000053"
PRIMARY_MATERIAL_PRIM = "/World/Looks/SteelCrack_000053"
INSPECTION_CAMERA = "/World/Cameras/CrackInspectionCamera"
SEMANTIC_LABEL = "steel_crack"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def look_at_matrix(eye, target, up=(0.0, 0.0, 1.0)):
    from pxr import Gf

    matrix = Gf.Matrix4d()
    matrix.SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up))
    return matrix.GetInverse()


def define_decal(
    stage,
    texture_path: str,
    prim_path: str = PRIMARY_DECAL_PRIM,
    material_path: str = PRIMARY_MATERIAL_PRIM,
    position=(0.0, 20.0, -4.005),
    size=(2.0, 1.5),
    rotation_degrees: float = 0.0,
):
    """Create a UV quad with a standards-compliant UsdPreviewSurface graph."""
    from isaacsim.core.utils.semantics import add_labels
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    stage.DefinePrim(DECAL_ROOT, "Xform")
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-0.5, -0.5, 0.0),
            Gf.Vec3f(0.5, -0.5, 0.0),
            Gf.Vec3f(0.5, 0.5, 0.0),
            Gf.Vec3f(-0.5, 0.5, 0.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, 0.0), Gf.Vec3f(0.5, 0.5, 0.0)])
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set(
        [
            Gf.Vec2f(0.0, 0.0),
            Gf.Vec2f(1.0, 0.0),
            Gf.Vec2f(1.0, 1.0),
            Gf.Vec2f(0.0, 1.0),
        ]
    )
    xform = UsdGeom.Xformable(mesh)
    translate_op = xform.AddTranslateOp()
    rotate_op = xform.AddRotateZOp()
    scale_op = xform.AddScaleOp()
    translate_op.Set(Gf.Vec3d(*position))
    rotate_op.Set(rotation_degrees)
    scale_op.Set(Gf.Vec3f(size[0], size[1], 1.0))

    material = UsdShade.Material.Define(stage, material_path)
    preview = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    preview.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.1)

    reader = UsdShade.Shader.Define(stage, f"{material_path}/PrimvarReader_st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    texture = UsdShade.Shader.Define(stage, f"{material_path}/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_path))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    texture.CreateOutput("a", Sdf.ValueTypeNames.Float)
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    preview.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(
        texture.ConnectableAPI(), "a"
    )
    preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh).Bind(material)

    prim = mesh.GetPrim()
    add_labels(prim, labels=[SEMANTIC_LABEL], instance_name="class")
    prim.SetCustomDataByKey("sourceDataset", "SteelCrack")
    prim.SetCustomDataByKey("sourceSplit", "Train")
    return {
        "mesh": mesh,
        "material": material,
        "texture": texture,
        "translate_op": translate_op,
        "rotate_op": rotate_op,
        "scale_op": scale_op,
    }


def set_texture(texture_shader, texture_path: str) -> None:
    from pxr import Sdf

    texture_shader.GetInput("file").Set(Sdf.AssetPath(texture_path))


def define_camera(
    stage,
    path: str,
    eye,
    target,
    focal_length: float = 35.0,
    aperture: float = 20.955,
):
    from pxr import Gf, UsdGeom

    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateHorizontalApertureAttr(aperture)
    camera.CreateVerticalApertureAttr(aperture)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 10000.0))
    xform = UsdGeom.Xformable(camera)
    transform_op = xform.AddTransformOp()
    transform_op.Set(look_at_matrix(eye, target))
    return camera, transform_op


def set_camera_pose(transform_op, eye, target) -> None:
    transform_op.Set(look_at_matrix(eye, target))


def inspection_camera_poses(target=(20.0, -5.995, 20.0), distance: float = 4.2) -> dict:
    poses = {}
    for name, angle_degrees in (("front", 0.0), ("left30", -30.0), ("right30", 30.0)):
        angle = math.radians(angle_degrees)
        poses[name] = (
            target[0] + math.sin(angle) * distance,
            target[1] + math.cos(angle) * distance,
            target[2],
        )
    return poses


def local_decal_corners(position, size, rotation_degrees: float) -> list[tuple[float, float, float]]:
    angle = math.radians(rotation_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    result = []
    for x, y in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
        scaled_x, scaled_y = x * size[0], y * size[1]
        result.append(
            (
                position[0] + cosine * scaled_x - sine * scaled_y,
                position[1] + sine * scaled_x + cosine * scaled_y,
                position[2],
            )
        )
    return result


def transfer_local_to_world(point) -> tuple[float, float, float]:
    """Apply EUN's TransferCrane translate(20,-10,0), rotateX(90) transform."""
    return (20.0 + point[0], -10.0 - point[2], point[1])


def world_decal_corners(position, size, rotation_degrees: float) -> list[tuple[float, float, float]]:
    return [
        transfer_local_to_world(point)
        for point in local_decal_corners(position, size, rotation_degrees)
    ]
