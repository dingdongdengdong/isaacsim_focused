#!/usr/bin/env bash
set -euo pipefail

container_name="eun-isaac51-steelcrack-evidence"
runtime_dir="/home/dong/eun/drone/.runtime/eun-webrtc"
scene_dir="/home/dong/eun/drone/eun_webrtc"
pegasus_dir="/home/dong/eun/drone/pegasus"
aerial_dir="/home/dong/eun/drone/aerial"
status_path="${runtime_dir}/evidence/steelcrack-000053/capture_status.json"

if docker container inspect "${container_name}" >/dev/null 2>&1; then
    echo "Container already exists: ${container_name}" >&2
    exit 2
fi

python3 - "${status_path}" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
if path.exists():
    path.unlink()
PY

docker run --detach --name "${container_name}" \
    --gpus all --network host --cpus 4 --shm-size 4g \
    --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_ENV_PRIVACY_CONSENT=1 \
    --mount "type=bind,src=${runtime_dir},dst=/workspace/run" \
    --mount "type=bind,src=${runtime_dir}/isaac_cache,dst=/isaac-sim/.cache" \
    --mount "type=bind,src=${scene_dir},dst=/workspace/eun_webrtc,readonly" \
    --mount "type=bind,src=${pegasus_dir},dst=/workspace/PegasusSimulator,readonly" \
    --mount "type=bind,src=${aerial_dir},dst=/workspace/aerial,readonly" \
    nvcr.io/nvidia/isaac-sim:5.1.0 \
    --/app/livestream/enabled=false \
    --/app/window/width=1280 --/app/window/height=720 \
    --exec /workspace/eun_webrtc/capture_scene_evidence.py >/dev/null

deadline=$((SECONDS + 900))
state=""
while (( SECONDS < deadline )); do
    if [[ -f "${status_path}" ]]; then
        state="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state", ""))' "${status_path}")"
        [[ "${state}" == "captured" || "${state}" == "failed" ]] && break
    fi
    [[ "$(docker inspect "${container_name}" --format '{{.State.Running}}')" == "true" ]] || break
    sleep 5
done
docker stop -t 10 "${container_name}" >/dev/null 2>&1 || true
exit_code="$(docker inspect "${container_name}" --format '{{.State.ExitCode}}')"
docker logs "${container_name}" >"${runtime_dir}/evidence/steelcrack-000053/capture_container.log" 2>&1 || true
docker container remove "${container_name}" >/dev/null
if [[ "${state}" != "captured" ]]; then
    echo "Evidence capture ended with state=${state:-timeout}, container_exit=${exit_code}" >&2
    exit 1
fi
cat "${status_path}"
