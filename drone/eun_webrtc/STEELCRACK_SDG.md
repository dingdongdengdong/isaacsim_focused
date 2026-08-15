# SteelCrack crane decal and SDG workflow

This EUN workflow uses only the SteelCrack `Train` split. `Validation` and
`Test` remain external evaluation surfaces and are never selected for Isaac
textures. DACL10K is not used.

## Runtime asset preparation

Generated source-derived pixels are stored outside source control:

```bash
python3 drone/eun_webrtc/prepare_steelcrack_assets.py
```

Default output:

```text
/home/dong/eun/drone/.runtime/eun-webrtc/crack_assets/
```

The directory contains 64 deterministic Train RGBA textures,
`selected_sources.json`, and the primary
`SteelCrack_000053.provenance.json`. Keep these artifacts internal until the
dataset redistribution/commercial-use terms have been confirmed with the
authors.

## WebRTC scene

`eun_scene.py` creates the primary decal at:

```text
/World/TransferCrane/CrackDecals/SteelCrack_000053
```

The mesh is a UV quad on the RTG main-girder front face. A
`UsdPreviewSurface` uses the RGBA texture's RGB and alpha channels with an
opacity threshold. The default WebRTC camera is the 4.2 m inspection camera;
the existing drone camera remains available through the existing camera
toggle.

## Synthetic data

Run and validate the required smoke set:

```bash
drone/eun_webrtc/generate_sdg.sh 100 steelcrack-smoke-100 20260815
python3 drone/eun_webrtc/validate_steelcrack_sdg.py \
  drone/.runtime/eun-webrtc/sdg/steelcrack-smoke-100
```

Only after the smoke report has `passed: true`, run the full set:

```bash
drone/eun_webrtc/generate_sdg.sh 5000 steelcrack-train-5000-v2 20260815
python3 drone/eun_webrtc/validate_steelcrack_sdg.py \
  drone/.runtime/eun-webrtc/sdg/steelcrack-train-5000-v2
```

Each manifest row records the frame and random seed, Train source ID, decal
pose and scale, camera pose/intrinsics, lighting, BasicWriter RGB/semantic
paths, and the homography mask path. Every fifth frame is a clean hard
negative. The validator decodes every output, creates the binary mask by
projecting the source alpha through the recorded planar homography, and checks
the projected decal bounds against the Replicator semantic output.

The camera remains inside the required 3-5 m range while its minimum distance
is constrained so that the complete rotated decal quad stays in frame. The
validator reports exact pixel IoU and also a one-pixel-tolerant IoU for thin
cracks affected by rasterization. Positive frames must be non-empty and the
one-pixel-tolerant IoU must be at least 0.85.

## Reviewed scene evidence

After the streaming scene has exported `scene.usda`, render the overview and
front/left/right inspection views with:

```bash
drone/eun_webrtc/capture_scene_evidence.sh
```

The PNGs and capture status are written to:

```text
/home/dong/eun/drone/.runtime/eun-webrtc/evidence/steelcrack-000053/
```

This is synthetic visual-detection evidence only. It is not structural-depth,
load-capacity, repair-authority, field-performance, or safety evidence.

## References

- [Isaac Sim 5.1 scene-based SDG](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/tutorial_replicator_scene_based_sdg.html)
- [UsdPreviewSurface](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/UsdPreviewSurface.html)
