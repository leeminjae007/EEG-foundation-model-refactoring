# 기본 geometry 마스킹 — 2026-09-12

사용자 요청에 따라 기본 사전학습 설정을 **GR9-1 모델 + 50% geometry tubelet 마스킹**으로 변경했습니다. `python pretrain.py`는 이제 `configs/pretrain.yaml`을 읽습니다. 기존 `configs/gr9_1.yaml`은 완료된 GR9-1의 I-JEPA 재현용으로 보존합니다.

후속 사용자 지시로 최종 기본값을 **target 50%, 연속 2–15초, REVE 공간 반경 3cm**로 정했습니다. 아래 GR2/GR6 성능 비교는 초기 선택 기록이며, 최종 조합이 최고 성능이라는 뜻은 아닙니다.

REVE 논문 Appendix A Table 5는 spatial masking radius를 3cm로 명시합니다. 공식 코드 commit `06a7059a07c3dabd80aee60c3dbc1eca4bdbe1c7`의 `src/configs/preprocessing/default.yaml`은 `radius_spat_mask: 0.03`을 쓰고 `src/utils/data_loading.py::spatial_masking`은 `KDTree.query_ball_point`로 3D 유클리드 이웃을 구합니다. [논문](https://papers.neurips.cc/paper_files/paper/2025/file/20a917f77773ac0fa8bea2bdd6606b66-Paper-Conference.pdf), [공식 설정](https://github.com/elouayas/reve_eeg/blob/06a7059a07c3dabd80aee60c3dbc1eca4bdbe1c7/src/configs/preprocessing/default.yaml), [공식 구현](https://github.com/elouayas/reve_eeg/blob/06a7059a07c3dabd80aee60c3dbc1eca4bdbe1c7/src/utils/data_loading.py#L499).

마스킹용 좌표는 MNE standard_1020의 미터 좌표로 별도 구합니다. SHPE용 정규화 좌표는 변경하지 않습니다. REVE에서 가져온 부분은 공간 반경과 거리 기준입니다. 전체 REVE 마스킹을 복제한 것은 아니며, 비율 50%·시간 2–15초·tubelet 합집합·최종 개수 절단은 이 프로젝트의 지정 방식입니다.

## 선택 근거

완료된 GR2-3, GR6-1, GR6-2의 각 14 task × seed 42, 1234, 696, 1001, 3407 결과 210개를 읽었습니다. 모든 결과가 설정된 전체 epoch를 완료했고 BAcc/R2로 선택됐는지 확인했습니다. Test 지표는 선택에 사용하지 않았습니다.

GR2-3의 binary task 5개는 CE, GR6의 채택된 binary task는 weighted CE이므로 세 모델의 해당 결과를 같은 조건으로 비교하지 않았습니다. 모델·데이터·optimizer 설정이 동일한 8개 분류 task의 task별 5-seed 검증 BAcc 평균을 다시 동일 가중치로 평균하여 선택했습니다. SEED-VIG의 R2는 별도 보고합니다. 이는 완료된 실험에서의 탐색적 선택이며 사전 등록된 기준은 아닙니다.

| 비교 지표 | GR2-3 | GR6-1 | GR6-2 |
|---|---:|---:|---:|
| 설정이 동일한 8개 분류 task 평균 validation BAcc | 0.587555 | 0.590089 | 0.597215 |
| SEED-VIG validation R2 | 0.461042 | 0.434374 | 0.428637 |
| 13개 분류 task 평균 validation BAcc (GR6 두 모델끼리만 동일 조건) | 0.661408 | 0.683033 | 0.692825 |

8개 분류 task는 TUEV, SEED-V, FACED, BCIC-IV-2a, PhysioNet-MI, BCIC2020-3, ISRUC, HMC입니다. GR6-2가 분류 평균에서 가장 높지만 SEED-VIG에는 GR2-3가 더 좋았습니다. 세 실험은 GR2 계열의 PE를 사용했으므로 GR9-1 PE에 적용한 새 조합의 성능은 아직 확인되지 않았습니다.

Task별 평균·SD, seed별 검증값과 210개 원본 result 경로·SHA256은 [geometry_selection.json](geometry_selection.json)에 있습니다. 재계산은 `python scripts/audit_geometry_selection.py /path/to/eeg-foundation-model`로 실행합니다. 기존 GR6의 stale CE 결과는 선택 자료에 포함하지 않았습니다.

## 가져온 알고리즘

설정 출처는 원본 `configs/gr6/pretrain_gr6-2_geometry_large.yaml`, 구현 출처는 `src/utils/masking.py::GeometryTubeletMaskingPolicy`입니다. 새 구현은 `src/modules/geometry_masking.py`에 있습니다.

1. 고정 TUEG 19채널 이름에 대응하는 MNE standard_1020의 실제 미터 좌표를 가져옵니다. EEG 값이나 개인별 좌표로 mask를 고르지 않습니다.
2. 중심 전극과 연속 2–15개의 1초 패치 및 시작 시각을 무작위로 고릅니다. 중심에서 3D 유클리드 거리 0.03m 이내 전극을 포함합니다.
3. 영역을 target 합집합에 추가합니다. 이미 target인 패치는 다시 추가하지 않습니다.
4. 마지막 영역이 목표 수를 넘으면 공간·시간 중심에 가까운 패치를 원본 점수와 tie-break 난수 순서로 골라 절단합니다. 따라서 마지막 tubelet은 완전한 원형·직사각형이 아닐 수 있습니다.
5. `round(570 × 0.5) = 285`개 target, 285개 context로 전체 격자를 분할합니다. 실제 target 비율은 50%입니다.
6. 모든 target을 하나의 decoder block 입력 집합으로 복원합니다. 기존 decoder layer 4개는 그대로 사용하며, 각 target 패치는 loss에 한 번만 기여합니다.

Tokenizer, encoder/decoder, PE, loss, optimizer, seed, global batch, epoch는 GR9-1과 같습니다. 마스킹은 새 학습 파라미터나 초기화 난수 소비를 추가하지 않습니다. 원본 geometry의 CPU torch seed → NumPy generator 순서를 유지합니다. 반경·기간·tubelet 수·마지막 영역 절단 횟수와 target/context 수를 학습 진단에 기록합니다.

## 실행과 기록 구분

```bash
# 새 기본값 (GPU 할당 후)
python -m torch.distributed.run --standalone --nproc_per_node=4 pretrain.py --distributed

# 기존 GR9-1 재현
python -m torch.distributed.run --standalone --nproc_per_node=4 pretrain.py \
  --config configs/gr9_1.yaml --distributed

# 새 기본값의 실제 데이터 1-update CPU 검증
python pretrain.py --device cpu --smoke
```

새 전체 학습 출력은 `outputs/pretrain_geometry50_reve3cm_t2_15`, CPU smoke 출력은 `outputs/smoke/pretrain_geometry50_reve3cm_t2_15`입니다. 다른 masking 설정의 checkpoint로 학습을 재개하면 가중치를 읽기 전에 오류를 냅니다. 기존 GR9-1 checkpoint를 재개하려면 해당 원래 설정을 명시해야 합니다.

`outputs/gr9_1_epoch40.pth`와 downstream 75개 설정은 기존 I-JEPA GR9-1 가중치를 계속 가리킵니다. 사용자가 최종 설정의 즉시 실험 제출을 승인했습니다. 제출 정보는 아래에 기록합니다. 기존 프로젝트·환경·학습 결과는 보존합니다.

## 변경 전 55% 검증 기록

CPU 작업 `27393239`가 정상 완료됐습니다. Python 3.8.16 / PyTorch 2.0.1 / FP32에서 9개 테스트가 41.48초에 통과했고 실제 TUEG batch 2의 optimizer 1회 update와 checkpoint 저장도 통과했습니다.

- 원본 geometry의 세 설정 × 다섯 seed × batch 3: mask, tubelet 진단값, RNG 상태 완전 일치. Target/context 분할 및 중복 제거 확인.
- 기본 마스킹 변경 전후: 모델 초기 tensor와 초기화 종료 RNG 완전 일치. 마스킹·새 출력 경로 외 설정 동일.
- 원본 GR9-1 가중치에 동일 geometry mask를 적용한 실제 EEG forward/backward: 출력, loss, 190개 parameter gradient 최대 절대 차이 모두 **0**, dropout 종료 RNG 일치.
- Target 314개, context 256개, prediction `[2,314,200]` 확인.
- 기존 I-JEPA 수치 동등성·checkpoint round trip·scheduler 회귀 테스트 통과.
- 다른 마스킹의 checkpoint로 재개하는 요청과 잘못된 geometry 입력을 거부하는 검사 통과.
- 실제 데이터 1-update smoke: 유한한 loss `0.7396193742752075`, optimizer 상태·유한한 모델 tensor·geometry 진단·새 경로 checkpoint 저장 확인. 이 값은 학습 성능 비교용 수치가 아닙니다.

[수치 보고서](../outputs/geometry_equivalence.json), [작업 로그](../outputs/geometry-check-27393239.log), [테스트](../tests/test_geometry.py)를 참조하세요. GPU BF16·DDP 동등성 및 새 조합의 전체 사전학습·downstream 성능은 검증하지 않았습니다. 과거 I-JEPA 검증은 `VALIDATION.md`에 별도로 유지합니다.


## 최종 설정 검증과 제출

최종 설정은 `geometry_tubelet`, `mask_ratio: 0.5`, `distance_metric: euclidean_m`, `radius_m: 0.03`, `min_time_patches: 2`, `max_time_patches: 15`입니다. 19×30 격자에서 target/context는 각각 285개입니다.

실제 standard_1020 19채널에서는 3cm 이내 이웃이 각 중심 전극 자신뿐입니다. 따라서 이 설정의 공간 영역은 한 전극이며 시간 구간을 합집합으로 모읍니다. 이는 물리 좌표와 REVE KDTree 이웃 기준을 직접 비교한 결과입니다. [거리·좌표·이웃 검증](../outputs/reve_radius_verification.json).

CPU 작업 `27393304`에서 테스트 **10개가 31.18초에 통과**했고 실제 TUEG batch 2의 optimizer 1회 update와 checkpoint 저장까지 완료했습니다. 새 마스크에서 기존 GR9-1 모델의 출력·loss·190개 gradient는 동결 원본과 최대 절대 차이 0, dropout RNG 상태도 일치했습니다. [수치 검증](../outputs/geometry_reve_equivalence.json), [로그](../outputs/reve-radius-check-27393304.log). 앞선 50%/5–15초/각도 반경 중간 설정의 테스트 작업 `27393282`는 5개 테스트를 통과했으나 제출 설정은 아닙니다.

사용자 승인으로 **Slurm 27393348**을 제출했습니다. Seed42로 새로 초기화하며 40 epoch, global batch512, 한 노드 A100 4개, CPU32, 메모리128GB, 제한시간24시간을 사용합니다. Pretrain warmup은 없고 기존 GR9-1의 AdamW/cosine을 유지합니다. 제출 확인 시 `PENDING / QOSMaxGRESPerUser`였습니다. 기존 작업이나 사용자 한도는 변경하지 않았습니다.

캠페인 디렉터리는 `outputs/geometry50_reve3cm_t2_15_20260912/`입니다. `source/`의 74개 파일을 SHA256으로 고정하고 읽기 전용으로 보존했으며 실제 작업은 이 복사본으로 실행합니다. 제출 후 기본 프로젝트 코드를 수정해도 이 작업의 코드나 설정은 바뀌지 않습니다. `training/`에 학습 결과를 기록합니다. CUDA 할당 후 학습 전에 동일 device·동일 캡처 RNG에서 GR9-1 공통 초기 tensor의 일치를 확인하고 `initialization_gpu.json`을 기록하도록 했습니다. 아직 대기 중이므로 CUDA 검증과 전체 학습 완료를 주장하지 않습니다.

[제출 원장](../outputs/geometry50_reve3cm_t2_15_20260912/submission.json), [고정 소스·설정 manifest](../outputs/geometry50_reve3cm_t2_15_20260912/manifest.json). `python scripts/submit_geometry_reve.py`를 다시 실행하면 기존 원장을 반환합니다. 최초 제출의 CLI `--nice 0` 구문 오류는 작업 생성 전 거절됐고, `--nice=0`으로 고친 뒤 제출했습니다. 거절 기록도 캠페인에 보존합니다.
