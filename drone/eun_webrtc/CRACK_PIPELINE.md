# 드론 RGB 카메라 균열 탐지 실행 가이드

이 문서는 `/home/dong/eun` 컴퓨터에서 다음 파이프라인을 재실행하는 방법을 설명한다.

```text
Pegasus Iris → 전면 하부 RGB 카메라 → ROS 2 Image
→ SteelCrack SegFormer-B0 → 마스크·신뢰도·결과 영상
```

디지털 트윈과 Replicator **데이터 생성**은 이 파이프라인의 범위가 아니다. 카메라는
드론 body prim의 자식으로 고정되며, WebRTC 기본 시점도 이 카메라를 사용한다.

## 준비 사항

- NVIDIA GPU와 Docker GPU runtime
- `nvcr.io/nvidia/isaac-sim:5.1.0` 이미지 접근 권한
- Pegasus checkout: `/home/dong/eun/drone/pegasus`
- Aerial dependency: `/home/dong/eun/drone/aerial`
- SteelCrack 원본: `/home/dong/ai/data/external/steelcrack`
- SegFormer 모델: `/home/dong/eun/runs/steelcrack-segformer-b0/best`
- AI Python: `/home/dong/ai/.venv/bin/python`

기본 경로가 없으면 `start.sh` 또는 `run_crack_pipeline.sh`가 즉시 실패한다.

## 1. Isaac Sim 시작

기존 `eun-isaac51-webrtc` 컨테이너가 있으면 상태를 먼저 확인한다.

```bash
docker ps -a --filter name=eun-isaac51-webrtc
```

소스가 변경되어 컨테이너를 재생성해야 할 때만 해당 컨테이너를 제거한다.

```bash
docker stop eun-isaac51-webrtc
docker rm eun-isaac51-webrtc
cd /home/dong/eun
./drone/eun_webrtc/start.sh
```

준비 완료는 다음 두 조건으로 확인한다.

```bash
cat drone/.runtime/eun-webrtc/status.json
docker logs eun-isaac51-webrtc 2>&1 | grep EUN_WEBRTC_READY
```

`status.json`의 `ready`가 `true`이고 TCP 49101이 열려 있어야 한다.

## 2. WebRTC 접속과 조종

Mac에서 SSH 터널을 유지한 상태로 Isaac Sim Streaming Client를 `127.0.0.1`에
연결한다.

```bash
ssh -N -L 49100:127.0.0.1:49101 dong@100.96.41.100
```

화면을 클릭해 키보드 포커스를 준 뒤 조종한다.

| 키 | 동작 |
| --- | --- |
| `W` / `S` | 전진 / 후진 |
| `A` / `D` | 좌 / 우 이동 |
| `R` / `F` | 상승 / 하강 |
| `Q` / `E` | 좌 / 우 yaw |
| `C` | 드론 카메라 / 외부 관찰 카메라 전환 |

주요 ROS 2 인터페이스는 다음과 같다.

```text
/drone0/cmd_vel       geometry_msgs/msg/Twist
/drone0/state/pose    geometry_msgs/msg/PoseStamped
/drone0/camera/rgb    sensor_msgs/msg/Image
```

## 3. RGB 캡처와 AI 추론

WebRTC 클라이언트를 연결한 상태에서 고유한 run 이름을 지정한다. 기존 run 결과는
덮어쓰지 않으므로 같은 이름을 재사용하면 실패한다.

```bash
cd /home/dong/eun
./drone/eun_webrtc/run_crack_pipeline.sh teammate-check-01
```

스크립트는 RGB 60장을 저장하면서 작은 전진 명령과 호버 명령을 같은
`/drone0/cmd_vel` 경로로 전송한 뒤 SegFormer 추론을 수행한다.

결과 위치:

```text
drone/.runtime/eun-webrtc/camera_stream/<run-name>/
drone/.runtime/eun-webrtc/pipeline_results/<run-name>/
```

주요 결과물:

```text
02_drone_flight_rgb.jpg
03_crane_crack_from_drone.jpg
04_ai_crack_detection.jpg
05_drone_to_crack_ai.mp4
detections.jsonl
manifest.json
```

`01_rgb_camera_mounted_drone.png`을 결과 폴더에 미리 넣은 경우 manifest에도 카메라
장착 증거로 포함된다. 이 파일은 실행 중 자동 생성하지 않는다.

## 설정 변경

필요한 값만 환경변수로 덮어쓴다.

| 변수 | 기본값 | 용도 |
| --- | --- | --- |
| `EUN_PUBLIC_ENDPOINT` | `100.96.41.100` | WebRTC 공개 주소 |
| `EUN_SIGNAL_PORT` | `49101` | WebRTC TCP 신호 포트 |
| `EUN_MEDIA_PORT` | `47999` | WebRTC UDP 미디어 포트 |
| `EUN_STEELCRACK_SOURCE` | `/home/dong/ai/data/external/steelcrack` | 원본 균열 데이터 |
| `EUN_CRACK_MODEL` | `runs/steelcrack-segformer-b0/best` | 추론 모델 |
| `EUN_AI_PYTHON` | `/home/dong/ai/.venv/bin/python` | AI 가상환경 Python |
| `EUN_AI_DEVICE` | `0` | CUDA 장치 또는 `cpu` |
| `EUN_CAPTURE_TIMEOUT_SECONDS` | `90` | RGB 캡처 제한 시간 |
| `EUN_CAPTURE_MAX_FRAMES` | `60` | 저장할 프레임 수 |
| `EUN_CAPTURE_FRAME_STRIDE` | `6` | ROS 수신 프레임 샘플 간격 |

예시:

```bash
EUN_AI_DEVICE=cpu EUN_CAPTURE_MAX_FRAMES=20 \
  ./drone/eun_webrtc/run_crack_pipeline.sh cpu-check-01
```

## 문제 해결

### WebRTC 화면이 검은색

```bash
docker ps --filter name=eun-isaac51-webrtc
ss -ltn | grep ':49101'
docker logs --tail 200 eun-isaac51-webrtc
```

- Streaming Client는 동시에 하나만 연결한다.
- 재접속이 안 되면 클라이언트를 완전히 종료한 후 다시 연결한다.
- `status.json`의 `camera_path`가 `/World/eun_iris/body/EunFpvCamera`인지 확인한다.

### RGB 토픽이 없거나 프레임이 오지 않음

Isaac Sim full-streaming은 WebRTC 클라이언트가 연결돼야 render product를 갱신한다.
새 런타임에서는 첫 render 이전까지 RGB publisher 자체가 DDS에 나타나지 않을 수도
있다. 클라이언트를 먼저 연결한 뒤 새 run 이름으로 다시 실행한다. 캡처는 기본
90초 후 명확한 오류로 종료되며 무한 대기하지 않는다.

### 모델 또는 Python 경로 오류

```bash
test -d /home/dong/eun/runs/steelcrack-segformer-b0/best
/home/dong/ai/.venv/bin/python -c 'import cv2, torch, transformers'
```

다른 위치를 사용하면 `EUN_CRACK_MODEL`과 `EUN_AI_PYTHON`을 지정한다.

## 개발 검증

```bash
cd /home/dong/eun
python3 -m unittest discover -s drone/eun_webrtc/tests -v
bash -n drone/eun_webrtc/start.sh drone/eun_webrtc/run_crack_pipeline.sh
git diff --check
```

비행 제어 테스트는 NumPy가 포함된 Isaac Sim Python에서 실행한다.

```bash
docker exec eun-isaac51-webrtc bash -lc \
  'cd /workspace/eun_webrtc && /isaac-sim/python.sh -m unittest test_flight_control_core.py -v'
```
