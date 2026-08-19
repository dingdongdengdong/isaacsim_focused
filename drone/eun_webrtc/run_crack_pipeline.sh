#!/usr/bin/env bash
set -euo pipefail

scene_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${scene_dir}/../.." && pwd)"
runtime_root="${EUN_RUNTIME_DIR:-${repository_root}/drone/.runtime/eun-webrtc}"
run_name="${1:-drone-crack-live}"
device="${EUN_AI_DEVICE:-0}"
ai_python="${EUN_AI_PYTHON:-/home/dong/ai/.venv/bin/python}"
capture_timeout="${EUN_CAPTURE_TIMEOUT_SECONDS:-90}"
max_frames="${EUN_CAPTURE_MAX_FRAMES:-60}"
frame_stride="${EUN_CAPTURE_FRAME_STRIDE:-6}"
capture_dir="${runtime_root}/camera_stream/${run_name}"
artifact_dir="${runtime_root}/pipeline_results/${run_name}"
capture_container="eun-rgb-capture-${run_name//[^a-zA-Z0-9_.-]/-}"
model_path="${EUN_CRACK_MODEL:-${repository_root}/runs/steelcrack-segformer-b0/best}"

cleanup_capture_container() {
    if docker container inspect "${capture_container}" >/dev/null 2>&1; then
        docker rm --force "${capture_container}" >/dev/null 2>&1 || true
    fi
}

if [[ ! "${run_name}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    echo "Run name must contain only letters, numbers, dot, underscore, and dash." >&2
    exit 1
fi
for numeric_value in "${capture_timeout}" "${max_frames}" "${frame_stride}"; do
    if [[ ! "${numeric_value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Capture limits must be positive integers: ${numeric_value}" >&2
        exit 1
    fi
done

if [[ "$(docker inspect eun-isaac51-webrtc --format '{{.State.Running}}' 2>/dev/null || true)" != "true" ]]; then
    echo "eun-isaac51-webrtc must be running before capture." >&2
    exit 1
fi
if docker container inspect "${capture_container}" >/dev/null 2>&1; then
    echo "Capture container already exists: ${capture_container}" >&2
    exit 1
fi
if [[ ! -e "${model_path}" ]]; then
    echo "Crack model is missing: ${model_path}" >&2
    exit 1
fi
if [[ ! -x "${ai_python}" ]]; then
    echo "AI Python is missing or not executable: ${ai_python}" >&2
    exit 1
fi
if find "${capture_dir}" "${artifact_dir}" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
    echo "Run output already exists; choose a new run name: ${run_name}" >&2
    exit 1
fi

mkdir -p "${capture_dir}" "${artifact_dir}"
chmod 0777 "${capture_dir}" "${artifact_dir}"

docker run --rm --detach \
    --name "${capture_container}" \
    --network host \
    --env ROS_DOMAIN_ID=0 \
    --env RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    --env FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    --mount "type=bind,src=${runtime_root},dst=/workspace/run" \
    --mount "type=bind,src=${scene_dir},dst=/workspace/eun_webrtc,readonly" \
    ros:jazzy-ros-base \
    bash -lc 'source /opt/ros/jazzy/setup.bash && python3 /workspace/eun_webrtc/ros2_rgb_capture.py --output-dir "/workspace/run/camera_stream/'"${run_name}"'" --max-frames '"${max_frames}"' --frame-stride '"${frame_stride}" \
    >/dev/null
trap cleanup_capture_container EXIT

# Preserve an initial hover segment in which the crack is centered. This also
# lets DDS discovery and the render product settle before control commands.
sleep 3

# Exercise the same ROS2 control path used by keyboard teleoperation while the
# RGB subscriber records. The deliberately tiny translation demonstrates that
# camera and vehicle move together without losing the inspection target.
docker run --rm --network host \
    --env ROS_DOMAIN_ID=0 \
    --env RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    --env FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    ros:jazzy-ros-base \
    bash -lc 'source /opt/ros/jazzy/setup.bash
set +e
timeout 1 ros2 topic pub --rate 10 /drone0/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.05, y: 0.0, z: 0.0}, angular: {z: 0.0}}"
move_status=$?
timeout 2 ros2 topic pub --rate 10 /drone0/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {z: 0.0}}"
hover_status=$?
if { [ "${move_status}" -ne 0 ] && [ "${move_status}" -ne 124 ]; } || { [ "${hover_status}" -ne 0 ] && [ "${hover_status}" -ne 124 ]; }; then
    exit 1
fi' \
    >/dev/null

set +e
capture_exit="$(timeout "${capture_timeout}" docker wait "${capture_container}")"
wait_status=$?
set -e
if [[ "${wait_status}" -eq 124 ]]; then
    echo "RGB capture timed out after ${capture_timeout}s. Connect the WebRTC client and retry with a new run name." >&2
    exit 1
fi
if [[ "${wait_status}" -ne 0 || "${capture_exit}" != "0" ]]; then
    echo "RGB capture failed (wait=${wait_status}, container=${capture_exit:-unknown})." >&2
    exit 1
fi

MPLCONFIGDIR=/tmp/eun-mpl \
    "${ai_python}" "${scene_dir}/detect_crack_stream.py" \
    "${capture_dir}" \
    --model "${model_path}" \
    --output-dir "${artifact_dir}" \
    --confidence 0.50 \
    --backend segformer \
    --device "${device}" \
    --fps 10

echo "capture_dir=${capture_dir}"
echo "artifact_dir=${artifact_dir}"
