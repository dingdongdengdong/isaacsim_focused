# EUN Isaac Sim WebRTC

이 폴더의 WebRTC 런타임은 기존 `aerial-ws-isaac51-webrtc`와 독립적으로 동작한다.

- 컨테이너: `eun-isaac51-webrtc`
- 이미지: `nvcr.io/nvidia/isaac-sim:5.1.0`
- 장면: `EUN Crane Crack Inspection`
- 로봇: 로컬 `pegasus/` 확장의 실제 Pegasus Iris
- 환경: 컨테이너 크레인과 RTG 크레인
- 균열: RTG 메인 거더의 카메라 방향 강재 면에 검은 코어와 붉은 테두리로 표시
- 신호 포트: TCP `49101`
- 미디어 포트: UDP `47999`
- Tailscale 서버 주소: `100.96.41.100`
- 상태 파일: `.runtime/eun-webrtc/status.json`
- PX4: 사용하지 않음
- ROS 2 제어: `/drone0/cmd_vel` → 속도 안정화 제어기 → rotor reference → Pegasus dynamics

균열은 WebRTC 시각 확인용 USD 형상이다. 실제 구조 건전성 판정이나 균열 탐지 모델의 추론 결과를 의미하지 않는다.

이 서버의 Isaac Sim 5.1 full-streaming `--no-window` 런타임에서는 별도 Replicator RGB probe가 빈 배열을 반환한다. 화면 확인은 Streaming Client에서 수행하며, 상태 파일의 `ready`는 장면·카메라·스트림 준비를 뜻하고 픽셀 검증을 뜻하지 않는다.

기존 드론 컨테이너의 TCP `49100`/UDP `47998`과 파일 또는 포트를 공유하지 않는다.

## ROS 2 WASD 비행과 드론 카메라

Streaming Client 화면을 클릭해 키보드 포커스를 준 뒤 다음 키를 사용한다.

| 키 | 동작 |
| --- | --- |
| `W` / `S` | 전진 / 후진 |
| `A` / `D` | 좌 / 우 이동 |
| `R` / `F` | 상승 / 하강 |
| `Q` / `E` | 좌 / 우 yaw |
| `C` | 외부 카메라 / 드론 탑재 카메라 전환 |

키보드 입력은 Isaac rigid body를 직접 옮기지 않는다. `geometry_msgs/msg/Twist`를
`/drone0/cmd_vel`에 발행하고, 속도·자세 안정화 제어기가 계산한 네 개 rotor RPM을
다음 ROS 2 토픽으로 전달한다.

```text
/drone0/control/rotor0/ref
/drone0/control/rotor1/ref
/drone0/control/rotor2/ref
/drone0/control/rotor3/ref
```

외부 ROS 2 노드는 동일한 `/drone0/cmd_vel` 토픽을 사용한다. 현재 Isaac Sim 5.1
컨테이너는 번들 ROS 2 Jazzy를 로드하므로, 외부 배포판에서 연결할 때는 DDS/RMW
호환성을 별도로 확인해야 한다.

```bash
ros2 topic pub --rate 10 /drone0/cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {z: 0.0}}"
```

상태 확인 토픽:

```bash
ros2 topic echo --once /drone0/state/pose
ros2 topic echo --once /drone0/state/twist
```

## 서버에서 시작

```bash
cd /home/dong/eun/drone
./eun_webrtc/start.sh
```

준비 상태 확인:

```bash
cat /home/dong/eun/drone/.runtime/eun-webrtc/status.json
docker logs eun-isaac51-webrtc 2>&1 | grep EUN_WEBRTC_READY
```

로그 추적:

```bash
docker logs -f eun-isaac51-webrtc
```

## Mac에서 접속

Isaac Sim Streaming Client 5.1은 기본 신호 포트 `49100`을 기대하므로, 기존 서버 포트와 충돌하지 않게 Mac의 로컬 `49100`을 EUN 서버의 `49101`로 전달한다.

Mac 터미널에서:

```bash
ssh -N -L 49100:127.0.0.1:49101 dong@100.96.41.100
```

이 터미널을 연 상태로 유지하고 Isaac Sim Streaming Client에서 서버 주소 `127.0.0.1`을 입력해 연결한다. 영상 미디어는 서버가 광고하는 Tailscale 주소 `100.96.41.100`과 UDP `47999`를 사용한다.

동시에 한 클라이언트만 연결한다. 연결이 끊긴 직후 재접속이 안 되면 클라이언트를 완전히 종료한 뒤 다시 연결한다.

## 중지와 재시작

```bash
docker stop eun-isaac51-webrtc
docker start eun-isaac51-webrtc
```

## 컨테이너 재생성

설정이나 장면 스크립트를 바꿨다면 기존 EUN 컨테이너만 제거한 뒤 다시 시작한다.

```bash
docker stop eun-isaac51-webrtc
docker rm eun-isaac51-webrtc
./eun_webrtc/start.sh
```

`aerial-ws-isaac51-webrtc` 컨테이너는 이 절차의 대상이 아니다.
