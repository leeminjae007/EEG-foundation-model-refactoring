# TUAB fine-tuning recovery

현재 nearest3–7 epoch40 weight를 고정하고, 완료된 warmup5 TUAB 5-seed 실행을 대조군으로 사용한다.
모든 epoch의 validation BAcc에서 best를 선택한다. AUROC selector도 유지한다.
초기 epoch를 선택 대상에서 제외하지 않는다.

## 제출할 20개 실행

| Array index | Arm | Warmup | Tokenizer / encoder LR | Head LR | Head | Head 선행 학습 |
|---|---|---:|---|---|---|---|
| 0–4 | `warmup0` | 0 | 1e-5 | 1e-5 | H=10 | 없음 |
| 5–9 | `backbone_lr_x0p1` | 0 | 1e-6 | 1e-5 | H=10 | 없음 |
| 10–14 | `head_first2_backbone_lr_x0p1` | 0 | 1e-6 | 1e-5 | H=10 | 2 epochs |
| 15–19 | `head_h4` | 0 | 1e-5 | 1e-5 | H=4 | 없음 |

각 arm의 seed 순서: **42, 1234, 696, 1001, 3407**.

- 공통: 총 20 epochs, batch 64 × accumulation 8, AdamW, WD 5e-5, head dropout 0.1, weighted binary CE, gradient clipping 1, BF16.
- 기본 cosine 최저 LR은 1e-6. Backbone LR 1/10 arm에서는 tokenizer/encoder 최저 LR도 1e-7로 낮춘다. Head 최저 LR은 1e-6이다.
- Head 선행 학습: epoch 1–2는 backbone `requires_grad=False`, `eval()`; epoch 3–20은 전체 미세조정. Head optimizer 상태와 전체 20-epoch cosine 진행은 유지한다. Backbone AdamW 상태는 첫 backbone update에서 생성된다.
- H=4는 `warmup0`와 비교한다. Backbone LR 감소나 head 선행 학습을 동시에 추가하지 않는다.
- Epoch 1–5와 6–20의 best validation BAcc를 함께 기록한다. Best epoch가 늦어졌다는 이유만으로 개선이라고 판단하지 않는다.

## SSH에서 한 번에 준비·검증·제출

아래 명령은 새 코드가 서버에 반영된 뒤 실행한다. 완료된 nearest3–7 campaign의 `array.json`, source manifest와 설정 snapshot, `result.json`, 현재 weight가 필요하다. 실제 baseline 설정은 array manifest가 가리키는 YAML에서 읽는다.

```bash
cd /gpfs/data/oermannlab/users/ml10266/workspace/EEG-founation-model
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m ablation.tuab.campaign all
```

이 명령은 baseline/weight 확인 → source/config snapshot → 실제 TUAB train 2 samples와 전체 모델을 이용한 CPU DDP 검증 → `sbatch` 순서로 진행한다.
Scientific test data는 smoke에서 읽지 않는다. 제출이 완료된 campaign을 다시 실행하면 기존 job ID를 출력한다.
제출 응답이 불명확하면 intent를 남기고 멈춘다. 이 경우 `sacct`로 확인하기 전 intent를 지우거나 다시 제출하지 않는다.

자원: 실행당 L40S 1 GPU, CPU 8, RAM 32 GB, 4시간. `gl40s_dev,gl40s_short,gl40s_long` 중 할당하며, 20개 전체를 array로 제출하고 최대 10개를 동시에 실행한다. 기존 TUAB 실제 소요 시간 약 1시간 39분을 기준으로 여유를 둔 walltime이다.

Array 프로세스마다 독립 rendezvous ID와 자동 할당 포트를 사용한다. [PyTorch 2.0.1의 같은 노드 내 여러 단일 노드 job 실행 방식](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/distributed/run.py)을 따른다.

```bash
# 제출된 job ID와 명령
cat outputs/tuab_recovery_20260915/submission.json

# 큐 상태
squeue -u "$USER" -n tuab-recovery-20260915 -o '%.18i %.12P %.10T %.10M %.6D %R'

# 제출 이력: batch/extern step 제외
sacct -X -u "$USER" -S 2026-09-15 --name=tuab-recovery-20260915 --format=JobID,JobName%30,State,ExitCode,Elapsed,NodeList

# validation/완료 결과 요약
.venv/bin/python -m ablation.tuab.report
```

## 구현과 검증

Campaign은 완료된 baseline의 `src` snapshot을 복사하고, 그 복사본에만 선택적 policy 인자, scheduler factory, epoch 시작 callback, DDP unused-parameter 설정을 연결한다. 원본 engine의 inline loop와 분리된 `train_finetune_epoch` 두 형태를 지원한다.
서버 원본 `src`는 수정하지 않는다. 기존 실행은 policy가 없으므로 기존 scheduler와 epoch 동작을 그대로 사용한다.

```bash
python -m pytest ablation/tests/test_tuab_finetuning.py ablation/tests/test_tuab_campaign.py tests/test_schedule.py -q
```

원래 engine과 기본 policy의 동일 결과, group LR 비율, CPU DDP freeze/unfreeze, AdamW 상태, epoch 1/2/3 중단 후 재시작과 최종 결과 일치를 검증한다.
서버 CPU smoke는 현재 실제 checkpoint·dataset·PyTorch 2.0.1 환경을 별도로 확인하며, 이를 통과해야 제출한다.

## 제출 기록 — 2026-09-15

- Job array: **27478589**, `0-19%10`, 총 20개.
- 제출 시각: 17:49:46 UTC / 13:49:46 EDT.
- 17:51:34 UTC 확인: 20개 모두 `PENDING (Priority)`, dependency 없음.
- 네 arm 모두 실제 TUAB train 데이터·전체 모델·CPU DDP·checkpoint 저장/복원 검증 통과. C는 epoch 1–2 frozen, 3–4 unfrozen을 실제 `train_finetune_epoch`로 검증했다.
- A/B/C 초기 전체 weight와 모든 arm의 backbone 초기 weight가 완료된 baseline의 `initialization.json`과 일치했다.
- 서버 원본 engine SHA256은 배포 전후 `87398166db49f9a7009ffab232b4dcc440a893e58ee702a0aefde9a884f8f3e6`로 동일하다.
- 권위 있는 제출 원장: `outputs/tuab_recovery_20260915/submission.json`.
