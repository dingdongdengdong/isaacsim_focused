#!/usr/bin/env bash
set -euo pipefail

container_name="eun-isaac51-webrtc"
scene_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${scene_dir}/../.." && pwd)"
runtime_dir="${EUN_RUNTIME_DIR:-${repository_root}/drone/.runtime/eun-webrtc}"
pegasus_dir="${EUN_PEGASUS_DIR:-${repository_root}/drone/pegasus}"
aerial_dir="${EUN_AERIAL_DIR:-${repository_root}/drone/aerial}"
steelcrack_source="${EUN_STEELCRACK_SOURCE:-/home/dong/ai/data/external/steelcrack}"
public_endpoint="${EUN_PUBLIC_ENDPOINT:-100.96.41.100}"
signal_port="${EUN_SIGNAL_PORT:-49101}"
media_port="${EUN_MEDIA_PORT:-47999}"
cpu_limit="${EUN_CPU_LIMIT:-8}"

for numeric_value in "${signal_port}" "${media_port}" "${cpu_limit}"; do
    if [[ ! "${numeric_value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Port and CPU values must be positive integers: ${numeric_value}" >&2
        exit 1
    fi
done

for required_path in "${pegasus_dir}" "${aerial_dir}" "${steelcrack_source}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path is missing: ${required_path}" >&2
        exit 1
    fi
done

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
    --source-root "${steelcrack_source}" \
    --output-dir "${runtime_dir}/crack_assets" \
    --source-id 000053 \
    --selection-count 64 >/dev/null
mkdir -p "${runtime_dir}/evidence"
chmod a+rwX "${runtime_dir}/crack_assets" "${runtime_dir}/evidence"

exec docker run --detach \
    --name "${container_name}" \
    --gpus all \
    --network host \
    --cpus "${cpu_limit}" \
    --shm-size 2g \
    --env ACCEPT_EULA=Y \
    --env PRIVACY_CONSENT=Y \
    --env OMNI_ENV_PRIVACY_CONSENT=1 \
    --env ROS_DOMAIN_ID=0 \
    --env RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    --env FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    --env EUN_ROS2_SELF_TEST="${EUN_ROS2_SELF_TEST:-0}" \
    --env EUN_PUBLIC_ENDPOINT="${public_endpoint}" \
    --env EUN_SIGNAL_PORT="${signal_port}" \
    --env EUN_MEDIA_PORT="${media_port}" \
    --mount "type=bind,src=${runtime_dir},dst=/workspace/run" \
    --mount "type=bind,src=${scene_dir},dst=/workspace/eun_webrtc,readonly" \
    --mount "type=bind,src=${pegasus_dir},dst=/workspace/PegasusSimulator,readonly" \
    --mount "type=bind,src=${aerial_dir},dst=/workspace/aerial,readonly" \
    nvcr.io/nvidia/isaac-sim:5.1.0 \
    --/app/livestream/publicEndpointAddress="${public_endpoint}" \
    --/app/livestream/port="${signal_port}" \
    --/app/livestream/fixedHostPort="${media_port}" \
    --/app/livestream/minHostPort="${media_port}" \
    --/app/livestream/maxHostPort="${media_port}" \
    --/app/livestream/allowDynamicResize=true \
    --/app/window/width=1280 \
    --/app/window/height=720 \
    --ext-folder /workspace/PegasusSimulator/extensions \
    --enable pegasus.simulator \
    --exec /workspace/eun_webrtc/eun_scene.py
