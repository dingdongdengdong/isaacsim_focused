#!/usr/bin/env bash
set -euo pipefail

frames="${1:-100}"
name="${2:-steelcrack-smoke-${frames}}"
seed="${3:-20260815}"
container_name="eun-isaac51-${name//[^a-zA-Z0-9_.-]/-}"
runtime_dir="/home/dong/eun/drone/.runtime/eun-webrtc"
scene_dir="/home/dong/eun/drone/eun_webrtc"
aerial_dir="/home/dong/eun/drone/aerial"
output_dir="${runtime_dir}/sdg/${name}"

if [[ -e "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Refusing to overwrite existing output: ${output_dir}" >&2
    exit 2
fi
if docker container inspect "${container_name}" >/dev/null 2>&1; then
    echo "Container already exists: ${container_name}" >&2
    exit 2
fi

python3 "${scene_dir}/prepare_steelcrack_assets.py" \
    --source-root /home/dong/ai/data/external/steelcrack \
    --output-dir "${runtime_dir}/crack_assets" \
    --source-id 000053 \
    --selection-count 64 >/dev/null
mkdir -p "${runtime_dir}/sdg"
mkdir -p "${runtime_dir}/isaac_cache"
chmod -R a+rwX "${runtime_dir}/crack_assets"
chmod a+rwx "${runtime_dir}/sdg" "${runtime_dir}/isaac_cache"

docker run --detach --name "${container_name}" \
    --gpus all \
    --network none \
    --cpus 4 \
    --shm-size 4g \
    --env ACCEPT_EULA=Y \
    --env PRIVACY_CONSENT=Y \
    --env OMNI_ENV_PRIVACY_CONSENT=1 \
    --env "EUN_SDG_FRAMES=${frames}" \
    --env "EUN_SDG_NAME=${name}" \
    --env "EUN_SDG_SEED=${seed}" \
    --mount "type=bind,src=${runtime_dir},dst=/workspace/run" \
    --mount "type=bind,src=${runtime_dir}/isaac_cache,dst=/isaac-sim/.cache" \
    --mount "type=bind,src=${scene_dir},dst=/workspace/eun_webrtc,readonly" \
    --mount "type=bind,src=${aerial_dir},dst=/workspace/aerial,readonly" \
    nvcr.io/nvidia/isaac-sim:5.1.0 \
    --/app/livestream/enabled=false \
    --/app/window/width=512 \
    --/app/window/height=512 \
    --exec /workspace/eun_webrtc/eun_sdg.py >/dev/null

deadline=$((SECONDS + 600 + frames * 15))
state=""
while (( SECONDS < deadline )); do
    if [[ -f "${output_dir}/status.json" ]]; then
        state="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state", ""))' "${output_dir}/status.json")"
        if [[ "${state}" == "generated" || "${state}" == "failed" ]]; then
            break
        fi
    fi
    if [[ "$(docker inspect "${container_name}" --format '{{.State.Running}}')" != "true" ]]; then
        break
    fi
    sleep 5
done
if [[ "${state}" != "generated" && "${state}" != "failed" ]]; then
    state="timeout"
fi
docker stop -t 10 "${container_name}" >/dev/null 2>&1 || true
exit_code="$(docker inspect "${container_name}" --format '{{.State.ExitCode}}')"
docker logs "${container_name}" >"${output_dir}/container.log" 2>&1 || true
docker rm "${container_name}" >/dev/null
if [[ "${state}" != "generated" ]]; then
    echo "SDG job ended with state=${state}, container_exit=${exit_code}; see ${output_dir}/container.log" >&2
    exit 1
fi
printf '%s\n' "${output_dir}"
