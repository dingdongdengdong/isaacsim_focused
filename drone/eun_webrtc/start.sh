#!/usr/bin/env bash
set -euo pipefail

container_name="eun-isaac51-webrtc"
runtime_dir="/home/dong/eun/drone/.runtime/eun-webrtc"
scene_dir="/home/dong/eun/drone/eun_webrtc"
pegasus_dir="/home/dong/eun/drone/pegasus"
aerial_dir="/home/dong/eun/drone/aerial"

if docker container inspect "${container_name}" >/dev/null 2>&1; then
    echo "Container ${container_name} already exists; refusing to replace it automatically." >&2
    exit 1
fi

mkdir -p "${runtime_dir}"
mkdir -p "${runtime_dir}/crane_assets"
chmod 0777 "${runtime_dir}/crane_assets"
mkdir -p "${runtime_dir}/viewport_debug"
chmod 0777 "${runtime_dir}/viewport_debug"
: > "${runtime_dir}/status.json"
chmod 0666 "${runtime_dir}/status.json"
python3 "${scene_dir}/prepare_steelcrack_assets.py" \
    --source-root /home/dong/ai/data/external/steelcrack \
    --output-dir "${runtime_dir}/crack_assets" \
    --source-id 000053 \
    --selection-count 64 >/dev/null
mkdir -p "${runtime_dir}/evidence"
chmod a+rwX "${runtime_dir}/crack_assets" "${runtime_dir}/evidence"

exec docker run --detach \
    --name "${container_name}" \
    --gpus all \
    --network host \
    --cpus 4 \
    --shm-size 2g \
    --env ACCEPT_EULA=Y \
    --env PRIVACY_CONSENT=Y \
    --env OMNI_ENV_PRIVACY_CONSENT=1 \
    --env ROS_DOMAIN_ID=0 \
    --env EUN_ROS2_SELF_TEST="${EUN_ROS2_SELF_TEST:-0}" \
    --mount "type=bind,src=${runtime_dir},dst=/workspace/run" \
    --mount "type=bind,src=${scene_dir},dst=/workspace/eun_webrtc,readonly" \
    --mount "type=bind,src=${pegasus_dir},dst=/workspace/PegasusSimulator,readonly" \
    --mount "type=bind,src=${aerial_dir},dst=/workspace/aerial,readonly" \
    nvcr.io/nvidia/isaac-sim:5.1.0 \
    --/app/livestream/publicEndpointAddress=100.96.41.100 \
    --/app/livestream/port=49101 \
    --/app/livestream/fixedHostPort=47999 \
    --/app/livestream/minHostPort=47999 \
    --/app/livestream/maxHostPort=47999 \
    --/app/livestream/allowDynamicResize=true \
    --/app/window/width=1280 \
    --/app/window/height=720 \
    --ext-folder /workspace/PegasusSimulator/extensions \
    --enable pegasus.simulator \
    --exec /workspace/eun_webrtc/eun_scene.py
