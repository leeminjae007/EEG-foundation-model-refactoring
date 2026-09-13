# 검증 범위와 결과

2026-09-12 기본 마스킹 변경: 새 기본값은 GR9-1 모델 + geometry 50%, REVE 반경 3cm, 시간 2–15초입니다. 아래 GR9-1 기록은 기존 `configs/gr9_1.yaml` 재현 기준이며, 변경 내용과 새 검증은 [GEOMETRY_DEFAULT.md](GEOMETRY_DEFAULT.md)를 참조하세요.

CPU, FP32, Python 3.8.16, PyTorch 2.0.1에서 검증했습니다. 학습 GPU를 사용하지 않았습니다. CPU 작업은 2 core·12 GB를 별도로 할당했으며, 전체 pretrain이나 downstream campaign을 제출하지 않았습니다.

## 수치 동등성

원본 epoch-40 checkpoint를 **strict load**하여 같은 실제 데이터, 같은 mask로 비교했습니다. 원본은 새 폴더의 동결된 테스트용 복사본을 별도 process에서 실행하므로 기존 프로젝트 Python 소스를 직접 import하지 않습니다.

| 검증 | 범위 | 최대 절대 차이 |
|---|---|---:|
| 초기화 | 같은 CPU RNG에서 새로 만든 모델 state 192 tensor | 0 |
| 사전학습 eval | 실제 TUEG `[2,19,6000]`, tokenizer·PE 적용값·12 axial block·4 decoder block·최종 출력 | 0 |
| 사전학습 loss | 원본 target 및 고정 I-JEPA mask, Smooth L1 beta 0.1 | 0 |
| 사전학습 train | Dropout 포함 출력·loss·190 parameter gradient | 0 |
| Downstream eval | 14 task 각각 실제 train sample 2개; ISRUC는 sequence 1개 | 0 |
| Downstream head 초기화 | 14 task head tensor의 SHA256 | 완전 일치 |
| Downstream gradient | SEED-V, MentalArithmetic, SEED-VIG | 0 |
| Loader | 14 task × train/val/test × 첫·마지막 sample, 총 84개 record | 완전 일치 |
| 전처리 | SEED-VIG 실제 recording 한 개의 앞 2개 window와 label | 완전 일치 |

허용 오차는 일반 output/gradient `atol=2e-6`, `rtol=1e-5`, loss `atol=1e-7`, `rtol=1e-6`으로 설정했으며 실제 차이는 전부 0이었습니다. 초기화와 loader는 허용 오차 0입니다. Stochastic 비교는 seed만 맞추지 않고 원본 forward 직전 torch RNG 상태를 복원하고 **forward/backward 후 RNG 상태까지 같은지** 확인했습니다.

세부 수치: [pretrain](../outputs/equivalence.json), [downstream](../outputs/downstream-equivalence.json), [dataset split 개수](../outputs/oracle-datasets.json). 테스트 6개는 Slurm `27392994`에서 79.06초에 통과했습니다. 해당 작업은 이어서 한-step 학습을 실행해 정상 완료했습니다. [로그](../outputs/verify-27392994.log).

## 학습·저장

- 실제 진입점 `pretrain.py`, `finetune.py`를 CPU `--smoke`로 실행했습니다. Batch 2, 한 optimizer update만 수행했으며 결과는 `outputs/smoke/`에 분리했습니다.
- Pretrain 및 SEED-V·MentalArithmetic·SEED-VIG에서 유한한 loss와 gradient, AdamW update, scheduler update, checkpoint 저장을 확인했습니다.
- Checkpoint round trip은 가중치를 실제로 변경한 뒤 strict load하여 복원했고 scheduler와 RNG 상태도 확인했습니다.
- Pretrain의 optimizer 이후 cosine 갱신, downstream의 optimizer 이전 warmup/cosine 갱신과 마지막 작은 accumulation group을 테스트했습니다.
- Classification 기본 평가의 threshold, BAcc/Kappa/F1, binary PR-AUC 정의, regression metric 수식을 유지했습니다. Validation/test는 중복 없는 stride sampler를 사용합니다.
- Attention 측정은 pre-dropout probability의 head별 entropy·self mass, valid pair의 row-centered QK/bias RMS, sampling coverage를 저장합니다. Temporal token이 하나이면 ratio 정의 여부를 따로 표시합니다. PE는 GELU/RMSNorm 이후 크기, gate는 값·pre-clip gradient를 기록합니다.
- Subject 진단에서 MentalArithmetic recording suffix를 사람 ID로 합치고, unknown ID를 한 사람으로 집계하지 않습니다. One-class 사람의 ranking metric은 ineligible로 기록하고 FP/FN count는 유지합니다. Probe는 person-class cell마다 최대 16개를 사용합니다.

최종 검증 5개도 Slurm `27393012`에서 29.13초에 통과했습니다. 따라서 **총 11개 검증 테스트가 통과**했습니다. Follow-up 변경 전 snapshot과 현재 snapshot의 output/loss/gradient 차이도 0이고 dropout RNG가 같았습니다. [소스 이력 비교](../outputs/source-history-equivalence.json), [네 학습 진입점의 한-step 결과와 strict-load](../outputs/smoke-summary.json), [최종 로그](../outputs/final-27393012.log).

추가로 각 8개 validation sample에서 분류·회귀 평가와 person-class probe를 실행했습니다. SEED-V 소량 subset에 없는 class를 예측했다는 sklearn 경고 1개는 그대로 표시했습니다. Metric 정의를 바꾸거나 경고를 숨기지 않았습니다. 이 소량 수치는 정식 성능 결과가 아닙니다. `pip check`와 Python compile 검사도 통과했습니다.

설치 실패 과정에서 남았던 중복 CUDA 11.7 wheel 패키지는 **새 환경에서만** 제거했습니다. 실제 실행에 쓰는 CUDA 11.8 native library는 독립 복사본으로 유지했습니다. 정리 후 PyTorch tensor 연산과 로드된 native library 경로를 [환경 검사](../outputs/environment-verification.json)에 기록했습니다.

## 보존 확인

[원본 파일 재검사](../outputs/original-preservation.json)에서 작업 시작 시 hash를 기록한 원래 소스·설정·스크립트의 변경은 0개입니다. 기존 checkpoint는 읽어서 복사했으며 원본으로 저장하지 않았습니다. 데이터는 기존 readonly loader를 사용합니다. 이번 작업의 CPU job 목록은 [검증 ledger](../outputs/verification_jobs.json)에 있습니다.

## 확인하지 못한 사항

- **CUDA BF16와 4-GPU DDP 수치 동등성/전체 학습 궤적은 미검증**입니다. 기존 GPU 실험을 방해하지 않도록 CPU만 사용했습니다. CUDA initialization을 CPU initialization과 같다고 해석하지 않습니다.
- **실행 당시 source-tree hash와 일치하는 완전한 소스 아카이브는 확인하지 못했습니다.** 현재 복사본과 follow-up 변경 전 복사본의 replay는 해당 두 snapshot의 동작을 확인하는 것이며, 2026-09-10 실행 당시 모든 파일의 동일성을 증명하지 않습니다.
- 모든 processed sample이나 전체 원시 전처리를 다시 실행하지 않았습니다. 확인한 loader sample은 split당 첫·마지막 두 개이고, split 크기·metadata·변환 tensor를 비교했습니다.
- `prepare_isruc_1.py`의 `edf_` 보조 모듈과 설정된 ISRUC 원시 데이터 경로를 찾지 못했습니다. BCIC-IV-2a 등 일부 기존 raw 경로도 현재 접근되지 않습니다. 이 코드를 다른 reader/filter로 대체하지 않았습니다. **ISRUC processed loader와 20-epoch sequence head의 실제 입력/출력은 검증했습니다.**
- 테스트 원본 소스와 checkpoint는 재현 oracle 용도로 `tests/reference*`에 남겼습니다. Production `src/`, `pretrain.py`, `finetune.py`는 이 코드를 import하지 않습니다.
- 과거 optimizer/checkpoint의 bitwise 학습 재개 변환은 제공하지 않습니다. 변환기는 가중치만 이식합니다. 새 형식의 저장·복원은 별도로 검증했습니다.
