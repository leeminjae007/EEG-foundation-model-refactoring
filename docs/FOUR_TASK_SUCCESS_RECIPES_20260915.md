# SEED-V · MentalArithmetic · ISRUC · HMC의 과거 고성능 설정

## 결론

네 task 모두 과거에 비교 논문 최고 보고 BAcc 평균을 넘긴 기록이 있다. 현재 nearest3–7 weight 결과만 보고 downstream hyperparameter가 잘못됐다고 판단하면 안 된다. 이전 결과의 **pretrain weight + head + optimizer + warmup + validation selector**를 함께 대조해야 한다.

아래는 기존 test 결과의 회고이다. 현재 weight에 채택할 설정은 동일 5-seed validation으로 판단하며, 서로 다른 pretrain 모델의 task별 test 최고값을 하나의 모델 성적으로 합치지 않는다. ‘보고 최고’의 범위는 기존 표의 CBraMod·CSBrain·REVE이며 통계적 유의성이나 최신 전체 문헌 SOTA를 뜻하지 않는다.

## 실제 result / resolved config를 대조한 5-seed 설정

Seeds: **42, 1234, 696, 1001, 3407**. Mean ± population SD, 모든 checkpoint는 validation BAcc로 선택했다.

| Task | 과거 weight / head | Validation BAcc | Test BAcc | 비교 논문 최고 BAcc | 현재 nearest W5 BAcc |
|---|---|---|---|---|---|
| SEED-V | GR9-2 epoch40 / H=4 | 0.3922 ± 0.0032 | **0.4203 ± 0.0035** | 0.4197, CSBrain | 0.4185 ± 0.0028 |
| MentalArithmetic | GR9-1 epoch40 / H=5 | 0.7057 ± 0.0224 | **0.8194 ± 0.0251** | 0.7660, REVE | 0.7097 ± 0.0451 |
| ISRUC | GR9-2 epoch40 / 전용 sleep head | 0.7706 ± 0.0086 | **0.7996 ± 0.0044** | 0.7925, CSBrain | 0.7932 ± 0.0022 |
| HMC | GR2-2 epoch40 / H=4 | 0.7353 ± 0.0022 | **0.7479 ± 0.0066** | 0.7401, REVE | 0.7365 ± 0.0041 |

공통 설정:

- Full fine-tuning, **50 epochs**, **warmup 0**.
- Tokenizer / encoder / head LR: **1e-4 / 1e-4 / 1e-4**.
- AdamW β=(0.9, 0.999), ε=1e-8, cosine minimum LR=1e-6, gradient clipping=1, BF16.
- Early stopping 없음. Train을 끝까지 진행하고 validation BAcc-best를 평가한다.
- 일반 head는 all-patch flatten → GELU MLP. H는 MLP hidden width / embedding dim이며 attention head 수가 아니다.

| Task | Batch × accumulation | WD | Head dropout | Label smoothing | Loss | Head 설정 |
|---|---|---|---|---|---|---|
| SEED-V | 64 × 1 | 0.01 | 0.1 | 0.1 | CE | `head_hidden_tokens: 4` |
| MentalArithmetic | 64 × 1 | 0.01 | **0.2** | YAML에는 0.1, binary loss에는 미적용 | Weighted binary CE | `null` → patch 수 5 |
| ISRUC | **2 × 32** | 0.01 | 0.1 | 0.1 | CE | 기존 전용 sleep head |
| HMC | 64 × 1 | **0.05** | 0.1 | 0.1 | CE | `head_hidden_tokens: 4` |

Historical config에 `warmup_epochs`가 없는 실행은 원본 downstream 코드의 `optimization.get('warmup_epochs', 0)`에 의해 0이다. 현재 binary loss도 label smoothing을 사용하지 않는다. MentalArithmetic의 0.1을 실제 binary smoothing 값으로 해석하지 않는다.

## Task별로 무엇이 달랐나

### SEED-V

GR9-2의 downstream HP는 현재 설정과 사실상 동일하며 warmup만 0 대 5이다. 하지만 GR9-2는 **spatial SH bias가 있는 다른 pretrain 모델**이므로 0.4203을 warmup 제거의 효과로 단정할 수 없다.

같은 GR9-1 weight에서 W0 / W3 / W5 test BAcc는 0.4173 / 0.4183 / 0.4154, validation BAcc는 0.3918 / 0.3915 / 0.3943이다. Test와 validation의 순위가 다르므로 task별 warmup을 test로 골라서는 안 된다. LR1e-4·WD0.01·dropout0.1·H4를 유지할 근거가 있고, 현재 best epoch36–43은 TUAB의 초반 하락과 다른 양상이다.

Notion에는 다음의 더 높은 선행 기록도 있다:

- **GR5-2: 0.4205 ± 0.0044, 5 seeds.** GR2-2에서 encoder/decoder SH 차원 비율만 100:100 / 50:50으로 변경한 모델. Downstream 설정은 유지했다는 실험 설명이 있다.
- **GR8-1: 0.4282 ± 0.0008, n=3.** Tokenizer/SHPE concat 뒤 identity-initialized linear mixer를 추가한 모델이다. 현재 보관된 Notion 표는 3-seed 부분 집계라 완성된 5-seed 결과로 표시하지 않는다.

이 두 캠페인의 실제 seed별 YAML과 완료 상태는 이번 SSH 차단으로 추가 대조하지 못했다. [과거 masking·SHPE·fusion 실험 표](https://www.notion.so/3d3d72c0a8e88196b16edfe849418c86)

### MentalArithmetic

GR9-1 W0의 **0.8194**는 같은 weight W5의 0.7674보다 높고, validation도 **0.7057 > 0.6920**이다. 같은 W5에서 dropout0.2→0.1로 낮춘 requested-recipe는 validation0.6849 / test0.7597로 더 낮았다.

따라서 **dropout0.2를 유지하고, 공통 warmup0 후보를 우선 검증**하는 근거가 있다. 현재 nearest W5 0.7097과 GR9-1 W0 0.8194 사이에는 pretrain weight와 warmup이 함께 달라졌다. 차이 전체를 한 요인으로 설명하지 않는다. H2 축소나 LR 감소는 이 task에서 이미 효과가 검증된 대체 설정이 아니다.

### ISRUC

실제 YAML을 대조한 GR9-2 W0는 **0.7996**, GR9-1 W0는 **0.7977**, GR9-1 W5는 0.7926이다. GR9-1의 validation도 W0 **0.7694 > W5 0.7612**로 W0가 높았다. **LR1e-4, WD0.01, dropout0.1, effective batch64(2×32), 전용 head**를 유지하는 방향이 맞다.

Notion 최신 GR9 표에는 **GR9-5 0.8040 ± 0.0079, 5 seeds**도 있다. 이 값은 실제로 더 높지만, GR9-5의 원본 resolved YAML과 checkpoint lineage는 이번 SSH 차단으로 추가 대조하지 못했다. 위 표의 검증된 GR9-2 설정을 GR9-5 설정이라고 바꿔 적지 않았다. [GR9 5-seed 결과](https://www.notion.so/3d7d72c0a8e8804c98e5d830fca7ddfc)

### HMC

같은 GR2-2 weight·W0·LR1e-4·WD0.05에서 **H30→H4**만 바꾼 비교:

| Head | Validation BAcc | Test BAcc | Weighted F1 | Kappa |
|---|---|---|---|---|
| H30 | 0.7324 ± 0.0012 | 0.7457 ± 0.0052 | 0.7653 ± 0.0058 | 0.7016 ± 0.0082 |
| H4 | **0.7353 ± 0.0022** | **0.7479 ± 0.0066** | 0.7624 ± 0.0075 | 0.6956 ± 0.0106 |

이번 BAcc 우선 목표에는 H4가 적합한 재검증 후보이다. 다른 두 지표의 하락도 보존한다. Head parameter는 145,207,205→19,362,005로 줄어든다. 현재 nearest weight에서 먼저 H4의 효과를 재검증해야 하며, 과거 개선폭을 그대로 더해 예측하지 않는다.

## Checkpoint와 원본 경로

원본 저장소 root: `/gpfs/data/oermannlab/users/ml10266/workspace/eeg-foundation-model`.
아래 trial directory의 `outputs/<trial>/checkpoint-epoch-0040.pth`를 사용했다.

| 표시 이름 | 원본 trial directory |
|---|---|
| GR9-1 | `gr9_1_equal_shpe_gelu_rmsnorm_d200_e40_20260910` |
| GR9-2 | `gr9_2_equal_shpe_gelu_rmsnorm_shbias_d200_e40_20260910` |
| GR2-2 | `gr3_2_smooth_l1_ijepa_multiblock_dual3_shlinear_d200_rmsnorm_e40_20260904` |

원본 캠페인:

- SEED-V / ISRUC GR9-2: `outputs/gr9_2_downstream_20260910/<trial>/<dataset>/eeg_mae/all_patch_reps/seed-<seed>/`.
- MentalArithmetic GR9-1: `outputs/gr9_1_downstream_20260910/<trial>/stress/eeg_mae/all_patch_reps/seed-<seed>/`.
- HMC H4: `outputs/downstream_capacity_warmup_20260906/03_hmc_h4/hmc_h4/<trial>/hmc/eeg_mae/all_patch_reps/seed-<seed>/`.
- HMC H30: `outputs/gr2_2_default_bacc_5seed/<trial>/hmc/eeg_mae/all_patch_reps/seed-<seed>/`.

수치와 실제 optimization/model 설정은 앞서 서버에서 수집한 `outputs/audits/downstream_hp_review_20260915/runs.json`의 seed별 원본 경로와 대조했다. 재집계 자료: `four_task_success_recipes.json`. Notion 두 페이지도 이번 작업에서 다시 조회했다.

## 현재 실행 범위

TUAB 4 arms × 5 seeds는 job **27478589**로 제출했다. [TUAB 실행 설정](../ablation/tuab/README.md).

후속 사용자 지시에 따라 위 네 task의 HP를 현재 nearest3–7 epoch40 weight에 적용한 **20개 실행을 job 27484155로 제출했다**(2026-09-15 19:02:10 UTC). LR1e-4, warmup0, 50 epochs를 공통으로 적용하고, 표의 WD·dropout·head·batch를 사용한다. Task별 과거 GR9-2/GR9-1/GR2-2 weight로 바꾸는 실험은 아니다. [네 task의 실행 설정](../ablation/four_task/README.md).

네 dataset 모두 실제 train 데이터·전체 모델·CPU DDP·checkpoint 저장/복원 검증을 통과했다. 19:03:37 UTC 현재 20개는 `PENDING (ReqNodeNotAvail)`이며 아직 학습 결과는 없다.
