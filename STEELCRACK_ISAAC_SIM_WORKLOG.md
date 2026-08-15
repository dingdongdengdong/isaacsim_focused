# SteelCrack 기반 Isaac Sim 크레인 크랙 작업 기록

## 완료 범위

- 데이터 소스는 로컬 SteelCrack `Train`만 사용했다. Validation/Test와 DACL10K는 텍스처 선택 및 합성 데이터 생성에서 사용하지 않았다.
- 기존 `Outline_*`/`Core_*` Cube 크랙을 제거하고 `/World/TransferCrane/CrackDecals/SteelCrack_000053` UV Mesh 데칼로 교체했다.
- `UsdPrimvarReader_float2 -> UsdUVTexture -> UsdPreviewSurface` 재질 그래프에 RGB/Alpha를 연결하고 `opacityThreshold=0.1`, `roughness=0.9`를 적용했다.
- 기본 WebRTC 시점을 4.2 m 정면 크랙 검사 카메라로 설정하고 기존 전체 장면/드론 카메라 경로는 유지했다.
- SteelCrack Train 64개 소스를 준비하고, 원본/마스크/생성 텍스처 SHA-256과 split/source ID를 provenance JSON에 기록했다.
- Isaac Sim 5.1 `BasicWriter`로 RGB, semantic segmentation, camera parameters를 생성하고, 원본 Alpha를 평면 homography로 투영한 binary mask와 교차 검증했다.
- 매 5번째 프레임을 clean hard negative로 만들었다.

## 주요 소스 파일

- `drone/eun_webrtc/prepare_steelcrack_assets.py`: Train-only RGBA와 provenance 생성
- `drone/eun_webrtc/steelcrack_usd.py`: 데칼/재질/카메라 USD 구성
- `drone/eun_webrtc/eun_scene.py`: WebRTC 크레인 장면과 상태 계약
- `drone/eun_webrtc/eun_sdg.py`: BasicWriter 합성 데이터 생성
- `drone/eun_webrtc/validate_steelcrack_sdg.py`: homography mask 및 semantic 정렬 검증
- `drone/eun_webrtc/capture_scene_evidence.py`: 저장된 실제 장면의 증거 PNG 렌더
- `drone/eun_webrtc/generate_sdg.sh`, `capture_scene_evidence.sh`: 실행 래퍼

## 런타임 검증 결과

### WebRTC 장면

- Isaac Sim 5.1 full streaming에서 장면 준비 완료를 확인했다.
- `status.json`의 `crack_mode=rgba_decal`, `source_dataset=SteelCrack`, `source_split=Train`, `source_id=000053`, texture SHA-256, decal/camera/semantic prim 필드를 확인했다.
- export된 `scene.usda`에서 `Outline_*`/`Core_*`가 0건이며 Mesh, `UsdUVTexture`, `UsdPreviewSurface`, Alpha 연결과 `opacityThreshold=0.1`을 확인했다.

### 100-frame smoke

- 경로: `drone/.runtime/eun-webrtc/sdg/steelcrack-smoke-100/`
- 결과: PASS, 100 frames = 80 positive + 20 clean
- 누락 파일, 비어 있는 positive mask, clean의 비정상 mask, 크기 불일치, Train 외 source 사용 없음
- 검토한 RGB에서 투명 사각형 없이 크랙 픽셀만 강재 표면에 표시됨

### 5,000-frame 최종 데이터셋

- 경로: `drone/.runtime/eun-webrtc/sdg/steelcrack-train-5000-v2/`
- 결과: PASS, 5,000 frames = 4,000 positive + 1,000 clean
- BasicWriter 산출물: 20,001 files, 약 646 MiB
- 정확 pixel IoU: 최소 `0.7827586207`, 평균 `0.9243993227`
- 1-pixel 허용 IoU: 최소 `0.8570552147`, 평균 `0.9471725409`
- 실패 0건. 얇은 크랙의 렌더러 rasterization 차이는 1-pixel 허용 IoU로도 별도 기록하며, positive/semantic mask가 비어 있으면 무조건 실패 처리한다.

첫 5,000-frame 시도는 큰 데칼을 3 m 근처에서 촬영할 때 UV 가장자리의 얇은 크랙이 잘리는 사례를 발견해 FAIL 처리했다. 회전된 전체 데칼이 화면 안에 들어오도록 3-5 m 범위 안에서 최소 카메라 거리를 계산하도록 수정한 후 v2를 재생성하고 통과시켰다.

## 직접 검토한 PNG

- 전체 장면: `drone/.runtime/eun-webrtc/evidence/steelcrack-000053/overview.png`
- 정면: `.../inspection_front.png`
- 좌 30도: `.../inspection_left30.png`
- 우 30도: `.../inspection_right30.png`
- 원본 binary mask: `.../binary_mask.png`
- 100-frame contact sheet: `.../sdg/steelcrack-smoke-100/validation_contact_sheet.png`
- 5,000-frame contact sheet: `.../sdg/steelcrack-train-5000-v2/validation_contact_sheet.png`

정면과 좌우 30도 PNG를 직접 확인했으며 데칼의 투명 영역이 사각형으로 표시되지 않았다. 저장된 PNG는 정적 시각 증거이며 동영상 기반 깜빡임 측정은 아니다.

## 데이터 및 사용 경계

- source-derived RGBA, USD export, RGB/mask/manifest는 `.runtime/` 아래에만 저장하며 Git에는 포함하지 않는다.
- SteelCrack 데이터 이용 조건이 별도 명확하지 않으므로 내부 비상업 연구로 제한하고, 재배포/상업 사용 전 저자 허가를 확인해야 한다.
- 이 결과는 합성 시각 탐지 증거다. 실제 SteelCrack Test 성능, 실제 크레인 촬영 성능, 균열 깊이/강도, 구조 안전 판정 또는 수리 권한의 증거가 아니다.

## 재현 명령

```bash
python3 -m unittest discover -s drone/eun_webrtc/tests -v
python3 -m py_compile drone/eun_webrtc/*.py

drone/eun_webrtc/generate_sdg.sh 100 steelcrack-smoke-100 20260815
python3 drone/eun_webrtc/validate_steelcrack_sdg.py \
  drone/.runtime/eun-webrtc/sdg/steelcrack-smoke-100

drone/eun_webrtc/generate_sdg.sh 5000 steelcrack-train-5000-v2 20260815
python3 drone/eun_webrtc/validate_steelcrack_sdg.py \
  drone/.runtime/eun-webrtc/sdg/steelcrack-train-5000-v2

drone/eun_webrtc/capture_scene_evidence.sh
```

## 참고 문서

- [Isaac Sim 5.1 scene-based SDG](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/tutorial_replicator_scene_based_sdg.html)
- [UsdPreviewSurface](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/UsdPreviewSurface.html)

## 완료 상태

- 작업 종료 시 실행 중인 Isaac Sim 컨테이너와 Kit 프로세스가 없음을 확인했다.
- `/home/dong/eun` Git 저장소의 `main` 브랜치에 코드, 테스트, 실행 문서와 이 작업 기록을 저장한다.
- `.runtime/`의 SteelCrack 파생 텍스처와 합성 데이터는 `.gitignore`로 제외한다.
