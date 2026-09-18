# Downstream hyperparameter 검토 — 2026-09-15

## 결론

TUAB는 전체 LR **1e-5**, WD **5e-5**, head dropout **0.1**, effective batch **512 (64×8)**, **20 epochs**, linear warmup **5 epochs**를 우선 권장한다. 현재 recipe를 유지할 근거가 있으며 warmup 5 자체의 우월성은 작다.
모든 task에 warmup 5를 적용하는 정책은 지지되지 않는다. 아래는 실제로 완료된 조합 가운데 GR9-1의 5-seed validation 평균을 우선한 권장 시작점이다. 현재 nearest 3–7채널 weight나 다른 encoder에서 확정한 최적값은 아니다.

## 자료와 비교 원칙

- 서버의 25개 campaign 경로에서 785개 result.json과 각각의 resolved_config.yaml을 읽었다. 이는 785개 독립 HP 조합 또는 785개 모두가 통제 비교라는 뜻이 아니다.
- 최근 통제 비교: GR9-1 기본 70개, warmup5 75개(TUAB LR 비교 5개 포함), warmup3 70개, requested recipe 30개, TUAB dropout/LR 및 speech LR 비교. 모든 권장 행은 같은 GR9-1 epoch-40 weight와 42/1234/696/1001/3407의 완료된 5개 seed에 근거한다.
- 분류는 validation BAcc-best, SEED-VIG는 validation R²-best. Test는 같은 checkpoint의 기록값이며 setting 선택 기준으로 사용하지 않는다. 표의 ±는 population SD이다.
- 서로 다른 pretrain weight를 합산하거나, AUROC/Kappa-selected 결과를 BAcc-selected 결과로 바꾸어 표기하지 않았다. 과거 seed43/636, early stopping, frozen/random, CE 비교는 해당 범위의 보조 근거로만 사용한다.
- 이 보고서는 과거 기록의 사후 검토이다. 작은 validation 차이를 통계적 유의성이나 전역 최적으로 해석하지 않는다. 같은 validation을 반복 탐색한 선택 편향도 남는다. [Cawley & Talbot, JMLR 2010](https://www.jmlr.org/beta/papers/v11/cawley10a.html)
- 기존 Notion의 “현재 fine-tune parameter defaults” 일부는 실제 실행값과 불일치한다. 예: TUAB WD는 실제 baseline 5e-5, TUEV dropout은 0.3, Mumtaz는 LR5e-5·WD0.05다. 뒤의 requested-recipe 비교표 및 실제 resolved_config를 우선했다.

- Source-tree SHA는 일부 동일 설정의 seed 사이에서도 다르다. 이 hash는 전체 configs/scripts/src/tests 및 최상위 문서를 포함하므로 차이를 학습 코드 변경으로 단정할 수 없다. 모든 과거 실행 코드의 byte 동일성까지 재검증한 것은 아니다. Warmup5 75개, warmup3 70개, requested recipe 30개는 저장된 initialization comparison에서 모두 공통 tensor 동일성을 통과했다.

## TUAB: 같은 GR9-1 weight에서 비교

아래는 모두 20 epochs, label smoothing 0, weighted CE, 동일한 5개 seed이다. WD 0.01 행은 batch까지 함께 바뀐 조합이므로 WD 단독 효과로 해석할 수 없다.

| 전체 LR | WD | Batch×accum | Dropout | Warmup epochs | Validation BAcc | Test BAcc | Test AUROC | Test AUPRC |
|---|---|---|---|---|---|---|---|---|
| 1e-05 | 5e-05 | 64×8 | 0.4 | 0 | 0.8300 ± 0.0027 | 0.8102 ± 0.0071 | 0.8882 ± 0.0036 | 0.8980 ± 0.0039 |
| 5e-06 | 5e-05 | 64×8 | 0.1 | 5 | 0.8323 ± 0.0006 | 0.8145 ± 0.0026 | 0.8893 ± 0.0007 | 0.9001 ± 0.0007 |
| 1e-05 | 5e-05 | 64×8 | 0.1 | 0 | 0.8325 ± 0.0033 | 0.8165 ± 0.0063 | 0.8910 ± 0.0025 | 0.9015 ± 0.0026 |
| 1e-05 | 5e-05 | 64×8 | 0.1 | 3 | 0.8309 ± 0.0009 | 0.8153 ± 0.0033 | 0.8893 ± 0.0021 | 0.9000 ± 0.0020 |
| 1e-05 | 5e-05 | 64×8 | 0.1 | 5 | 0.8329 ± 0.0010 | 0.8139 ± 0.0031 | 0.8892 ± 0.0019 | 0.9001 ± 0.0017 |
| 3e-05 | 5e-05 | 64×8 | 0.1 | 0 | 0.8235 ± 0.0036 | 0.7996 ± 0.0071 | 0.8771 ± 0.0098 | 0.8861 ± 0.0100 |
| 1e-05 | 0.01 | 256×1 | 0.1 | 5 | 0.8321 ± 0.0043 | 0.8170 ± 0.0029 | 0.8903 ± 0.0025 | 0.9006 ± 0.0025 |

- **LR 3e-5는 제외 우선순위가 높다.** Dropout 0.1·warmup0를 고정했을 때 GR9-1 validation BAcc 0.8325→0.8235, test 0.8165→0.7996. GR2-2와 GR9-2도 같은 방향이다.
- **LR 5e-6로 내릴 근거는 약하다.** Warmup5 고정 시 validation 0.8329→0.8323. Test의 작은 상승만으로 교체하지 않는다.
- **Dropout 0.1이 현재 GR9 계열의 우선값이다.** GR9-1은 validation 0.8300→0.8325, test 0.8102→0.8165. Test 개선은 GR2-2·GR9-2에서도 보이지만 GR2-2 validation은 0.8366→0.8325로 반대였다. 모든 weight에서 validation이 개선됐다고 주장하지 않는다.
- **Warmup 5와 0은 TUAB에서 근접하다.** Dropout0.1에서 validation 차이는 약 0.0004. Warmup5의 seed SD가 작아 현재값을 유지하는 판단이며 test 최고값이라는 뜻은 아니다.
- **Batch256·WD0.01은 추가 후보로 보존한다.** Test BAcc는 높지만 validation 평균이 낮고 SD는 크다. Batch 변경으로 epoch당 optimizer update 횟수도 달라진다.
- **Warmup2의 과거 개선은 별도 조건이다.** GR2-2, unweighted CE, dropout0.4에서 seed42 screening + 나머지4seed confirmation이었다. Dropout0.1·weighted CE·warmup2 조합을 검증했다고 합성해서 말할 수 없다.
- Warmup5/LR1e-5의 validation 선택 epoch는 3–4였다. 이것은 3–4 epoch만 학습하라는 뜻이 아니다. 20-epoch cosine horizon을 유지하고 validation으로 checkpoint를 선택한다.

## 14개 task 권장 설정

LR은 tokenizer·encoder·head에 동일하게 적용한다. Batch는 GPU microbatch×gradient accumulation이며 1 GPU 기준이다. 표의 dropout은 downstream head dropout이다.

| Dataset | LR | WD | Head dropout | Label smoothing | Warmup epochs | Epochs | Batch×accum | Head hidden tokens |
|---|---|---|---|---|---|---|---|---|
| TUAB | 1e-05 | 5e-05 | 0.1 | 0 | 5 | 20 | 64×8 | 기존 기본값 |
| TUEV | 0.0001 | 0.01 | 0.3 | 0.1 | 0 | 50 | 64×1 | 기존 기본값 |
| CHB-MIT | 0.0001 | 0.01 | 0.1 | 0 | 5 | 20 | 64×1 | 기존 기본값 |
| SEED-V | 0.0001 | 0.01 | 0.1 | 0.1 | 5 | 50 | 64×1 | 4 |
| SEED-VIG | 0.0001 | 0.01 | 0.1 | 0 | 5 | 50 | 64×1 | 기존 기본값 |
| FACED | 0.0001 | 0.01 | 0.1 | 0.1 | 0 | 50 | 64×1 | 기존 기본값 |
| Mumtaz | 0.0001 | 0.01 | 0.1 | 0 | 5 | 50 | 64×1 | 기존 기본값 |
| MentalArithmetic | 0.0001 | 0.01 | 0.2 | 0.1 | 0 | 50 | 64×1 | 기존 기본값 |
| BCIC-IV-2a | 0.0001 | 0.01 | 0.2 | 0.1 | 3 | 50 | 64×1 | 기존 기본값 |
| PhysioNet-MI | 5e-05 | 0.01 | 0.3 | 0.1 | 0 | 50 | 64×1 | 기존 기본값 |
| BCIC2020-3 | 5e-05 | 0.01 | 0.1 | 0.1 | 0 | 50 | 64×1 | 기존 기본값 |
| ISRUC | 0.0001 | 0.01 | 0.1 | 0.1 | 0 | 50 | 2×32 | SleepModel |
| HMC | 0.0001 | 0.05 | 0.1 | 0.1 | 5 | 50 | 64×1 | 기존 기본값 |
| Siena | 0.0001 | 0.05 | 0.1 | 0 | 0 | 50 | 64×1 | 기존 기본값 |

공통: AdamW β=(0.9,0.999), ε=1e-8, cosine minimum LR1e-6, gradient clip1, BF16, full fine-tuning, all-patch GELU head. Binary(TUAB/CHB-MIT/Mumtaz/MentalArithmetic/Siena)는 train-count 기반 weighted CE, multiclass는 CE, SEED-VIG는 MSE. Binary의 실제 구현은 1-logit weighted BCE 계열이다. No early stopping; validation primary 선택은 유지한다.
기존 head_hidden_tokens=null은 일반 task에서 입력의 시간 patch 수로 해석된다. SEED-V는 명시적 H=4이며 ISRUC는 별도 SleepModel이다. Head 축소 실험의 H는 attention head 수가 아니다.

## 선택한 조합의 실제 결과와 판단

각 행의 수치는 같은 GR9-1 pretrained checkpoint에서 해당 설정으로 이미 실행된 기록이다. 서로 다른 최적 단일 요인을 합성한 미실행 조합이 아니다.

| Dataset | Validation primary | Test primary | 판단 |
|---|---|---|---|
| TUAB | 0.8329 ± 0.0010 | 0.8139 ± 0.0031 | LR 1e-5·dropout 0.1 유지. Warmup 5와 0의 validation 차이는 작아 5의 우월성 확정은 아님. |
| TUEV | 0.6579 ± 0.0247 | 0.6874 ± 0.0096 | Warmup 0이 validation·test 모두 우세. Dropout 0.3 유지. |
| CHB-MIT | 0.8356 ± 0.0452 | 0.8031 ± 0.0696 | Warmup 5/3는 근접. 5는 validation 평균 우세, 3은 test와 seed 변동에서 유리한 재검증 후보. |
| SEED-V | 0.3943 ± 0.0019 | 0.4154 ± 0.0058 | Warmup 5가 validation에서 우세하나 test 개선은 없음. H=4 유지. |
| SEED-VIG | 0.4680 ± 0.0122 | 0.1419 ± 0.0218 | Warmup 5가 GR9-1에서 validation R²·test R² 모두 우세. 기존 head 유지. |
| FACED | 0.6340 ± 0.0073 | 0.5996 ± 0.0046 | Warmup 0이 validation·test 모두 우세. |
| Mumtaz | 0.9675 ± 0.0137 | 0.9033 ± 0.0094 | LR 1e-4 + WD 0.01 + warmup 5 묶음의 5-seed 개선. LR·WD 개별 효과는 분리 불가. 원본 split 중복 한계 있음. |
| MentalArithmetic | 0.7057 ± 0.0224 | 0.8194 ± 0.0251 | Dropout 0.2 + warmup 0 권장. 평균 우세지만 seed별 방향은 일관되지 않아 중간 수준 근거. |
| BCIC-IV-2a | 0.5408 ± 0.0059 | 0.5128 ± 0.0114 | Validation 우선: dropout 0.2 + warmup 3. Dropout 0.1 + warmup 5도 별도의 유력 후보. |
| PhysioNet-MI | 0.5988 ± 0.0042 | 0.6533 ± 0.0023 | LR 5e-5 + dropout 0.3 + warmup 0이 validation 평균 최고. LR 1e-4 + dropout 0.1 + warmup 5와 차이는 작음. |
| BCIC2020-3 | 0.4877 ± 0.0147 | 0.4883 ± 0.0172 | 전체 LR 5e-5 + warmup 0. GR2-2·GR9-1·GR9-2에서 지지, GR9-3에서는 역효과. |
| ISRUC | 0.7694 ± 0.0080 | 0.7977 ± 0.0102 | Warmup 0이 validation·test 모두 우세. 실제 microbatch 2 × accumulation 32 유지. |
| HMC | 0.7360 ± 0.0007 | 0.7422 ± 0.0029 | Warmup 0/3/5 validation 차이 0.0006 미만. 기존 warmup 5 유지하되 최적값 미확정. |
| Siena | 0.9516 ± 0.0062 | 0.8624 ± 0.0187 | Warmup 0이 validation·test 모두 우세. Weighted CE 유지. |

Primary는 BAcc, SEED-VIG만 R²다.

### Warmup 비교 전체

TUAB는 dropout0.1·LR1e-5로 맞춰 warmup0 행을 읽었다. 다른 task는 기존 task 설정을 유지한 warmup0/3/5 비교다. 숫자는 5-seed 평균이다.

| Dataset | Val W0 | Val W3 | Val W5 | Test W0 | Test W3 | Test W5 |
|---|---|---|---|---|---|---|
| TUAB | 0.8325 | 0.8309 | 0.8329 | 0.8165 | 0.8153 | 0.8139 |
| TUEV | 0.6579 | 0.6412 | 0.6269 | 0.6874 | 0.6836 | 0.6655 |
| CHB-MIT | 0.7821 | 0.8333 | 0.8356 | 0.7740 | 0.8330 | 0.8031 |
| SEED-V | 0.3918 | 0.3915 | 0.3943 | 0.4173 | 0.4183 | 0.4154 |
| SEED-VIG | 0.4374 | 0.4579 | 0.4680 | 0.1047 | 0.1293 | 0.1419 |
| FACED | 0.6340 | 0.6235 | 0.6164 | 0.5996 | 0.5863 | 0.5834 |
| Mumtaz | 0.9507 | 0.9429 | 0.9500 | 0.8991 | 0.9024 | 0.8967 |
| MentalArithmetic | 0.7057 | 0.7022 | 0.6920 | 0.8194 | 0.7292 | 0.7674 |
| BCIC-IV-2a | 0.5250 | 0.5408 | 0.5342 | 0.4425 | 0.5128 | 0.4821 |
| PhysioNet-MI | 0.5988 | 0.5942 | 0.5933 | 0.6533 | 0.6494 | 0.6537 |
| BCIC2020-3 | 0.4845 | 0.4637 | 0.4595 | 0.4829 | 0.4869 | 0.4859 |
| ISRUC | 0.7694 | 0.7667 | 0.7612 | 0.7977 | 0.7921 | 0.7926 |
| HMC | 0.7357 | 0.7354 | 0.7360 | 0.7434 | 0.7440 | 0.7422 |
| Siena | 0.9516 | 0.9400 | 0.9439 | 0.8624 | 0.8487 | 0.8373 |

### 해석이 갈리는 task

- **CHB-MIT:** W5 validation 0.8356±0.0452, W3 0.8333±0.0279로 차이가 작다. Test는 W5 0.8031±0.0696, W3 0.8330±0.0326이다. Validation 평균 우선이면 W5 유지, 안정성 재검증 후보는 W3. 현재 nearest checkpoint의 W5 결과와 GR9-1 W3를 섞어서 비교하면 안 된다.
- **BCIC-IV-2a:** W5·dropout0.2 → W5·dropout0.1에서 validation 0.5342→0.5385, test 0.4821→0.5385로 개선됐다. 다만 W3·dropout0.2의 validation은 0.5408로 조금 더 높다. 따라서 두 조합을 남기고, 미실행 W3·dropout0.1을 확정 최적으로 제시하지 않는다.
- **PhysioNet-MI:** W5에서 LR5e-5·dropout0.3 → LR1e-4·dropout0.1은 validation 0.5933→0.5974, test 0.6537→0.6564. 그러나 기존 LR5e-5·dropout0.3·W0 validation이 0.5988로 더 높다. 새로운 묶음도 유력하지만 superiority는 작고 두 요인이 동시에 바뀐다.
- **SEED-V:** W5는 W0보다 5개 seed 모두 validation이 좋아졌지만 test 평균은 0.4173→0.4154로 낮아졌다. 검증셋 선택 기준에 따라 유지하는 설정이지 test 개선이 확인된 설정은 아니다.
- **Mumtaz:** 새 묶음은 W5 baseline 대비 validation 0.9500→0.9675(4/5 seed 상승), test 0.8967→0.9033이다. 이전 저LR·WD0.05 단일-seed 결과보다 최근 같은-weight 5-seed 비교를 우선한다. 다만 공개 원본 데이터에 train–validation 및 validation–test 중복 신호가 확인되어 이 설정의 외부 일반화 근거는 제한된다.
- **HMC:** Validation W0/W3/W5 = 0.7357/0.7354/0.7360. SD보다 작은 차이이므로 실질적으로 우열 미확정이다. W5를 유지하는 이유는 현재 기준 설정이기 때문이다.

## 이전 비교군에서 얻은 추가 근거

### BCIC2020-3 LR 비교

전체 tokenizer·encoder·head LR만 1e-4에서 5e-5로 변경했다. Warmup0, WD0.01, dropout0.1, label smoothing0.1, batch64, 50epochs를 유지한 각 5-seed 비교다.

| Pretrain | LR1e-4 test BAcc | LR5e-5 test BAcc | LR1e-4 validation BAcc | LR5e-5 validation BAcc |
|---|---|---|---|---|
| GR2-2 | 0.4909 ± 0.0057 | 0.5147 ± 0.0111 | 0.4992 ± 0.0162 | 0.5253 ± 0.0120 |
| GR9-1 | 0.4829 ± 0.0110 | 0.4883 ± 0.0172 | 0.4845 ± 0.0175 | 0.4877 ± 0.0147 |
| GR9-2 | 0.4856 ± 0.0271 | 0.5013 ± 0.0190 | 0.4749 ± 0.0298 | 0.4901 ± 0.0168 |
| GR9-3 | 0.4976 ± 0.0099 | 0.4821 ± 0.0088 | 0.4789 ± 0.0132 | 0.4664 ± 0.0096 |

GR9-3에서는 LR5e-5가 validation과 test 모두 하락했다. 따라서 encoder·pretrain과 무관한 보편적 최적값은 아니다. 별도 GR2-2 seed42에서 tokenizer/encoder5e-5·head1e-4가 validation0.4907→0.5480, 잠금 후 test0.4893→0.5453이었다. 하지만 한 seed이며 전체LR5e-5의 5-seed 실험과 다른 조합이므로 추가 후보로만 둔다.

### Backbone LR, head 크기, 기존 논문 recipe

- BCIC-IV-2a GR2-2 3-seed(42/43/696) 탐색: backbone5e-5·head1e-4가 validation0.5249→0.5272로 선택됐다. Test BAcc 평균은 둘 다0.4838, SD0.0426→0.0095였다. 안정화 신호이지 평균 향상이 아니며 현재 5-seed와 합산하지 않는다.
- SEED-V GR2-2 seed42: backbone1e-4→5e-5는 validation0.3997→0.3966. SEED-VIG도 backbone 절반 및 warmup5 단일-seed에서 validation R²가 기존0.5039보다 낮았다. 반면 GR9-1의 5-seed SEED-VIG는 W5 개선이 확인되므로 weight 의존성이 있다.
- Small-head GR2-2: SEED-VIG H4 test R²0.1298→0.1193, speech H2 BAcc0.4909→0.4885. HMC H4는 BAcc0.7457→0.7479지만 F1·Kappa가 감소했다. 현재 새 weight에서 head 축소를 자동 적용할 만큼 일관된 근거가 없다. CHB·MentalArithmetic의 당시 head 비교는 unweighted CE로 현재 weighted CE와 조건이 다르다.
- 과거 CSBrain HP 묶음을 같은 GR1-3 weight에 적용한 FACED(seed42)는 LR5e-4·WD0.1·dropout0.3에서 validation/test BAcc가 모두0.1111로 붕괴했다. ISRUC의 LR1e-3·effective batch32 조합도 test0.7703이었다. 논문별 recipe를 우리 backbone에 그대로 옮긴다고 최적이 되지 않는다.
- MentalArithmetic 과거 GR4 WD0.1→0.01 비교는 test0.7382→0.7500이지만 validation0.6195→0.6138이었다. 따라서 WD0.01 자체가 validation으로 증명된 보편적 최적값이라는 표현은 피한다. 현재 GR9-1 비교는 WD0.01을 고정한 warmup·dropout 비교다.
- Frozen/random 및 validation threshold 변경은 transfer/decision-protocol 비교다. 여기의 full-finetuning·기존 threshold 설정 추천과 합치지 않았다. BCIC-IV-2a Balanced Softmax는 train class count가 동일해 기존 CE와 동등하므로 별도 개선 요인으로 세지 않는다.

## 권장 조합의 세 가지 test metrics

| Dataset | Metric | Mean ± SD |
|---|---|---|
| TUAB | balanced_accuracy | 0.8139 ± 0.0031 |
| TUAB | auroc | 0.8892 ± 0.0019 |
| TUAB | auprc | 0.9001 ± 0.0017 |
| TUEV | balanced_accuracy | 0.6874 ± 0.0096 |
| TUEV | weighted_f1 | 0.8336 ± 0.0113 |
| TUEV | kappa | 0.6767 ± 0.0211 |
| CHB-MIT | balanced_accuracy | 0.8031 ± 0.0696 |
| CHB-MIT | auroc | 0.9024 ± 0.0346 |
| CHB-MIT | auprc | 0.3748 ± 0.0980 |
| SEED-V | balanced_accuracy | 0.4154 ± 0.0058 |
| SEED-V | weighted_f1 | 0.4215 ± 0.0078 |
| SEED-V | kappa | 0.2719 ± 0.0095 |
| SEED-VIG | pearson | 0.5278 ± 0.0108 |
| SEED-VIG | r2 | 0.1419 ± 0.0218 |
| SEED-VIG | rmse | 0.2942 ± 0.0037 |
| FACED | balanced_accuracy | 0.5996 ± 0.0046 |
| FACED | weighted_f1 | 0.6034 ± 0.0042 |
| FACED | kappa | 0.5483 ± 0.0051 |
| Mumtaz | balanced_accuracy | 0.9033 ± 0.0094 |
| Mumtaz | auroc | 0.9698 ± 0.0102 |
| Mumtaz | auprc | 0.9750 ± 0.0076 |
| MentalArithmetic | balanced_accuracy | 0.8194 ± 0.0251 |
| MentalArithmetic | auroc | 0.9141 ± 0.0257 |
| MentalArithmetic | auprc | 0.8288 ± 0.0565 |
| BCIC-IV-2a | balanced_accuracy | 0.5128 ± 0.0114 |
| BCIC-IV-2a | weighted_f1 | 0.4906 ± 0.0151 |
| BCIC-IV-2a | kappa | 0.3505 ± 0.0153 |
| PhysioNet-MI | balanced_accuracy | 0.6533 ± 0.0023 |
| PhysioNet-MI | weighted_f1 | 0.6543 ± 0.0020 |
| PhysioNet-MI | kappa | 0.5377 ± 0.0030 |
| BCIC2020-3 | balanced_accuracy | 0.4883 ± 0.0172 |
| BCIC2020-3 | weighted_f1 | 0.4883 ± 0.0173 |
| BCIC2020-3 | kappa | 0.3603 ± 0.0215 |
| ISRUC | balanced_accuracy | 0.7977 ± 0.0102 |
| ISRUC | weighted_f1 | 0.8155 ± 0.0050 |
| ISRUC | kappa | 0.7607 ± 0.0090 |
| HMC | balanced_accuracy | 0.7422 ± 0.0029 |
| HMC | weighted_f1 | 0.7593 ± 0.0042 |
| HMC | kappa | 0.6939 ± 0.0050 |
| Siena | balanced_accuracy | 0.8624 ± 0.0187 |
| Siena | auroc | 0.9222 ± 0.0136 |
| Siena | auprc | 0.3506 ± 0.1216 |

## 근거와 재현

- [진짜최종: TUAB dropout/LR, speech LR, GR9 비교](https://www.notion.so/3d7d72c0a8e8804c98e5d830fca7ddfc)
- [Requested recipe·warmup3 완료 기록](https://www.notion.so/3dbd72c0a8e881f38cd2fd8ef6d0207c)
- [GR2-2 validation tuning·transfer 기록](https://www.notion.so/3d3d72c0a8e8812b89cbd633aceac1b1)
- [Capacity·warmup·loss 과거 분석](https://www.notion.so/3d3d72c0a8e88196b16edfe849418c86)
- [GR1/GR3/GR4 및 과거 recipe 기록](https://www.notion.so/3c9d72c0a8e880eb9ffdc9ec45bed506)
- 로컬 audit: `outputs/audits/downstream_hp_review_20260915/runs.json`, `supplemental.txt`, `recommendations.json`. 각 run의 원격 result/config 경로, optimizer/head 설정, checkpoint 경로, seed, epochs_ran, selector별 validation/test가 들어 있다.
- 분석 재현: `E:/workspace/.venv-ablation/Scripts/python.exe outputs/audits/downstream_hp_review_20260915/report.py`.
- 검증: 모든 권장 조합은 동일한 5개 seed, 같은 checkpoint 경로, 요청한 전체 epoch 완료, 실제 동일 LR 세 그룹임을 확인했다. 가중치가 다른 row를 합산하지 않았다.
