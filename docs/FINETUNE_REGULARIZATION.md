# 세 소규모 task의 fine-tuning 과적합 조정

GR9-1 warmup5의 완료된 Mumtaz, MentalArithmetic, BCIC-IV-2a × 5 seed 결과를 확인했습니다. 각 실행의 50개 epoch train/validation 기록을 읽었으며 test 지표는 설정 결정에 사용하지 않았습니다.

| Task | 마지막 train BAcc 평균 | 마지막 validation BAcc 평균 | 선택된 validation BAcc 평균 | Best epoch 범위 |
|---|---:|---:|---:|---:|
| Mumtaz | 1.0000 | 0.9056 | 0.9500 | 6–19 |
| MentalArithmetic | 1.0000 | 0.6538 | 0.6920 | 3–14 |
| BCIC-IV-2a | 1.0000 | 0.5139 | 0.5342 | 6–20 |

각 값은 seed 42, 1234, 696, 1001, 3407의 평균입니다. 선택된 검증값은 각 run의 validation BAcc-best checkpoint 값입니다. 마지막 epoch 값과 구분합니다. [전체 곡선·원본 경로·hash](../outputs/finetune_regularization_20260912/audit.json).

세 task 모두 train BAcc는 100%에 도달하지만 validation은 훨씬 낮고 선택 epoch 이후 개선되지 않거나 하락합니다. 특히 MentalArithmetic은 다섯 seed 중 세 개의 best epoch가 3–4입니다. 이 양상은 과적합과 일치하지만, 피험자 간 분포 차이도 함께 작용할 수 있으므로 gap만으로 원인을 특정하지 않습니다. Mumtaz의 선택된 validation BAcc는 95.0%로, 다른 두 task와 심각도가 같지는 않습니다.

현재 all-patch MLP head도 큽니다. Head 파라미터는 Mumtaz 19,201,401개, MentalArithmetic 20,201,401개, BCIC-IV-2a 14,241,804개입니다. 따라서 사전학습된 표현의 지나친 변화와 head의 높은 용량을 별도로 점검합니다.

## 첫 실행: tokenizer·encoder LR 0.1배

| Task | Tokenizer LR | Encoder LR | 유지하는 head LR |
|---|---:|---:|---:|
| Mumtaz | 5e-6 | 5e-6 | 5e-5 |
| MentalArithmetic | 1e-5 | 1e-5 | 1e-4 |
| BCIC-IV-2a | 1e-5 | 1e-5 | 1e-4 |

사전학습된 tokenizer와 encoder의 업데이트 속도만 낮추는 하나의 요인입니다. Head dropout, head 크기, weight decay, batch, loss, 50 epoch, warmup5, min LR 1e-6, all-patch GELU full fine-tuning을 보존합니다. Binary weighted CE와 multiclass CE를 유지하며, validation BAcc와 AUROC/Kappa checkpoint를 같은 run에서 선택합니다. Early stopping은 추가하지 않습니다.

최종 checkpoint를 validation-best로 이미 선택하므로 epoch 수만 줄여서는 선택 모델이 개선된다고 볼 수 없습니다. 이번 비교는 학습 경로 자체를 바꾸며, 세 task 각각의 완전한 5-seed validation BAcc 평균으로 평가합니다. Gap 감소만으로 성공으로 판정하지 않습니다.

## 별도로 준비한 후속 후보

첫 실행과 섞지 않은 다음 두 후보도 task × 5 seed로 준비했습니다. 아직 실행 대상으로 채택하지 않았습니다.

- `head_dropout_plus0p2`: Mumtaz 0.1→0.3, MentalArithmetic와 BCIC-IV-2a 0.2→0.4. LR과 head 크기는 baseline 유지.
- `head_h2`: hidden token 수를 2로 줄임. Mumtaz와 MentalArithmetic은 5→2, BCIC-IV-2a는 4→2. Head 파라미터는 Mumtaz 7,680,801개, MentalArithmetic 8,080,801개, BCIC-IV-2a 7,121,404개. LR과 dropout은 baseline 유지.

총 45개 후보 YAML은 `configs/finetune_regularization/`에 있습니다. 첫 실행은 LR 후보 15개만 사용합니다. 기존 기준 설정은 `configs/downstream/`에 보존합니다. Binary 구현에서 `label_smoothing`은 적용되지 않으므로 해당 값만 올리는 후보는 만들지 않았습니다.

새 geometry pretrain은 별도 실험이며 아직 이 fine-tuning의 source checkpoint가 아닙니다. 여기서는 완료된 기존 I-JEPA GR9-1 epoch40 가중치를 사용합니다. 원본 warmup3 campaign과도 결과를 섞지 않습니다. 새 실행 엔진과 원본 엔진의 과거 CPU 동등성은 확인됐지만, GPU 학습 전체의 bit-exact 일치는 검증된 바 없으므로 원본 실행과의 비교는 탐색적으로 해석합니다.

## 실행 기록

앞선 즉시 제출 지시를 이어 첫 LR 후보 **15개를 Slurm 배열 `27393566` (`0-14%3`)으로 제출**했습니다. 세 task × seed42/1234/696/1001/3407이며, 한 작업당 L40S 1개·CPU8·메모리32GB·24시간, 동시 실행 최대3개입니다. 나머지 두 후보는 준비 상태입니다.

CPU 검증 작업 `27393550`이 정상 완료했습니다. 제출할 고정 소스와 설정으로 세 task의 실제 데이터 batch2를 각각 한 번 학습하고, 유한한 모델 가중치, optimizer 상태, checkpoint 저장과 tokenizer/encoder/head optimizer group의 LR 비율 0.1:0.1:1을 확인했습니다. [검증 결과](../outputs/finetune_regularization_20260912/smoke_validation.json).

실제 실행은 `outputs/finetune_regularization_20260912/source/`의 읽기 전용 소스와 설정을 사용합니다. `source_manifest.json`에는 소스 및 GR9-1 변환 checkpoint의 hash를 기록했습니다. 기존 프로젝트와 기준 설정, 제출된 geometry 사전학습은 수정하지 않았습니다.

[제출 원장](../outputs/finetune_regularization_20260912/submission.json), [15개 실행 manifest](../outputs/finetune_regularization_20260912/lr_array.json). `python scripts/submit_regularization_lr.py`를 다시 실행하면 기존 원장을 반환하며 중복 제출하지 않습니다.

`python scripts/report_regularization_lr.py`로 task별 완료 상태를 갱신할 수 있습니다. 각 task의 5개 seed가 모두 완료된 후에만 기준 대비 validation BAcc 평균 차이를 계산하며 test 지표는 비교에 사용하지 않습니다. [현재 비교 상태](../outputs/finetune_regularization_20260912/validation_comparison.json). 아직 학습 결과가 없어 과적합 완화나 성능 개선을 확인한 상태는 아닙니다.
