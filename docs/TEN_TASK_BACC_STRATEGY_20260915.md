# 10개 downstream BAcc 개선 전략 — 2026-09-15

## 반영한 사용자 방향

- 대상: TUAB, TUEV, CHB-MIT, SEED-V, FACED, MentalArithmetic, PhysioNet-MI, ISRUC, HMC, Siena.
- 제외: BCIC-IV-2a, BCIC2020-3, SEED-VIG, Mumtaz.
- Task별 warmup 선택은 하지 않는다. 최신 지시에 따라 warmup 0도 허용하며 공통 0 대 공통 5를 비교한다.
- 목표는 10개 task의 BAcc 개선이다. 현재 표의 목표값은 기존 비교군 CBraMod·CSBrain·REVE의 최고 보고 평균이며, 전체 최신 문헌의 글로벌 SOTA를 확정한 표가 아니다.
- 하나의 공통 pretrained checkpoint를 고정한 다음 downstream 방법을 비교한다. 서로 다른 weight의 task별 최고값을 하나의 모델 결과로 합치지 않는다.

## 현재 위치

서버 확인: 2026-09-15T14:21:58.067042Z. 현재 nearest 3–7채널 pretrain epoch40 weight의 결과다. 9개 task는 5개 seed 완료, Siena는 2개 seed 결과만 있다. 표의 ±는 population SD이며 단순 평균 초과와 통계적 우월성은 다르다.

| Task | 현재 BAcc | 비교군 최고 BAcc | 차이 (percentage points) | 우선순위 |
|---|---|---|---|---|
| TUAB | 0.8166 ± 0.0028 | 0.8315 (REVE) | -1.49 | 높음: 초반 이후 성능 저하 |
| TUEV | 0.6972 ± 0.0033 | 0.6903 (CSBrain) | +0.69 | 현재 성능 유지 |
| CHB-MIT | 0.8350 ± 0.0107 | 0.7398 (CBraMod) | +9.52 | 현재 성능 유지·warmup 변경 시 보호 |
| SEED-V | 0.4185 ± 0.0028 | 0.4197 (CSBrain) | -0.12 | 낮은 폭의 개선 필요 |
| FACED | 0.5827 ± 0.0031 | 0.5752 (CSBrain) | +0.75 | 현재 성능 유지 |
| MentalArithmetic | 0.7097 ± 0.0451 | 0.7660 (REVE) | -5.63 | 최우선: gap과 과적합 모두 큼 |
| PhysioNet-MI | 0.6535 ± 0.0072 | 0.6480 (REVE) | +0.55 | 현재 성능 유지 |
| ISRUC | 0.7932 ± 0.0022 | 0.7925 (CSBrain) | +0.07 | 근소 초과·안정성 확인 |
| HMC | 0.7365 ± 0.0041 | 0.7401 (REVE) | -0.36 | 작은 head 우선 |
| Siena | 0.8468 ± 0.0109 (n=2) | 0.7662 (CSBrain) | +8.06 | 나머지 seed 완료 확인 |

기준 수치 출처: [기존 비교표](https://www.notion.so/3dbd72c0a8e881f7be8ecdc27aa0e918), 서버 archived published_references.json. TUAB·HMC의 REVE 수치는 [NeurIPS 2025 원문](https://papers.neurips.cc/paper_files/paper/2025/file/20a917f77773ac0fa8bea2bdd6606b66-Paper-Conference.pdf)의 Table 16·14에서도 확인했다.

## TUAB: 초반 선택의 실제 원인 후보

5개 seed의 BAcc-best epoch는 42=2, 1234=3, 696=1, 1001=2, 3407=3이다. Warmup이 5여도 모두 1–3 epoch에서 선택된다.

![TUAB validation and sampled training curves](../outputs/audits/downstream_hp_review_20260915/tuab_learning_curves.png)

음영은 5개 seed의 SD, 회색 영역은 warmup이다. Train loss는 100 optimizer update 간격으로 기록한 batch loss의 평균이며 전체 epoch loss가 아니다. 재시작으로 중복 저장된 validation epoch는 마지막 기록 하나를 사용했다.

| Epoch | Validation BAcc 평균 | Validation AUROC 평균 | 기록된 train batch loss 평균 |
|---|---|---|---|
| 1 | 0.8193 | 0.9056 | 0.5580 |
| 3 | 0.8230 | 0.9019 | 0.3411 |
| 5 | 0.8147 | 0.8991 | 0.2680 |
| 10 | 0.8062 | 0.8894 | 0.1703 |
| 20 | 0.8012 | 0.8856 | 0.0693 |

관찰: train loss는 계속 감소하지만 validation BAcc와 AUROC가 함께 악화된다. 현재 현상은 후반 과적합과 일치한다. Pretrained 표현의 훼손, 큰 head의 암기, 분포 차이가 각각 얼마나 기여하는지는 이 곡선만으로 확정할 수 없다.
Warmup 제거만으로 해결될 근거도 없다. 과거 GR9-1의 warmup0·dropout0.1에서도 best epoch가 1–3이었다. Epoch5 이전 checkpoint를 선택에서 제외하면 저장 시점만 바뀌고 성능 저하는 남는다.
성공 기준은 validation 최고 BAcc 자체의 상승과 후반 성능 저하의 감소다. 현재 TUAB 각 seed의 epoch1–5 최고 validation BAcc 평균은 0.82435, epoch6–20 최고 평균은 0.81694다. 같은 구간의 차이를 진단하되 모든 epoch에서 validation-best를 선택한다.

## 공통 warmup 판단

이전 GR9-1 weight에서 대상 10개만 모아 비교했다. TUAB는 두 arm 모두 dropout0.1·LR1e-5로 맞췄다.

| 공통 warmup | 10-task validation BAcc 평균 | 10-task test BAcc 평균 |
|---|---|---|
| 0 | 0.7060 | 0.7171 |
| 5 | 0.7032 | 0.7074 |

Warmup0가 validation에서 6/10, test에서 7/10 task의 평균이 높다. 따라서 공통 warmup0를 첫 비교 후보로 추천한다. 다만 CHB-MIT validation은 W0 0.7821 대 W5 0.8356, test는 0.7740 대 0.8031로 반대다. 평균 한 개로 10개 모두의 개선을 대신 판단하지 않는다. 이 수치는 GR9-1 기록이며 현재 nearest weight에서 warmup0의 효과는 아직 미검증이다.

## 첫 실험: head와 backbone의 학습 속도 분리

먼저 TUAB와 MentalArithmetic에서 아래 순서로 한 요인씩 비교한다. Current W5는 완료된 기준 결과를 보존한다. 비교는 현재 nearest pretrained weight·고정 split·기존 head·loss·batch·dropout·epochs를 유지한다.

| Arm | Warmup | Head LR | Tokenizer·encoder LR | 바꾸는 요인 |
|---|---|---|---|---|
| 기준 | 5 | 기존값 | 기존값 | 완료 결과 |
| A | 0 | 기존값 | 기존값 | Warmup 제거 |
| B | 0 | 기존값 | Head의 0.1배 | Backbone 학습 속도 감소 |
| C | 0 | 기존값 | Head의 0.1배 | B + 초기 2 epoch head만 학습 |

- TUAB B/C: head1e-5, tokenizer1e-6, encoder1e-6. TUAB 기존의 모든 group LR을 함께5e-6로 줄인 실험과 다르다.
- MentalArithmetic B/C: head1e-4, tokenizer1e-5, encoder1e-5.
- 최소 LR도 group 비율에 맞춘다: head1e-6, tokenizer/encoder1e-7. TUAB backbone base LR1e-6에 공통 minimum1e-6을 그대로 쓰면 warmup 이후 cosine decay가 없어지므로 이 경우를 명시적으로 피한다.
- C의 첫 2 epoch는 backbone requires_grad=False 및 eval 모드, head는 train 모드다. 이후 backbone을 학습 모드로 전환해 전체 fine-tune한다. Head optimizer 상태를 유지하고 backbone optimizer 상태는 첫 실제 update에서 시작한다. Warmup 추가나 scheduler 재시작은 하지 않는다.
- 총 epoch는 TUAB20(2+18), MentalArithmetic50(2+48)로 유지한다. Head 준비 2 epoch를 제외하고 full-finetune의 best 시점과 누적 update 수도 비교해야 한다. 날짜상 epoch가 2만큼 늦어진 것만으로 성공이라고 판정하지 않는다.
- Random head와 backbone을 동시에 적응시키면 pretrained 표현이 변형될 수 있다는 [LP-FT 연구](https://arxiv.org/abs/2202.10054)가 동기다. 여기서는 기존 nonlinear MLP head를 사용하므로 정확한 linear-probing 재현이 아니라 그 아이디어를 적용한 head-first fine-tuning 후보다. EEG에서의 개선은 미검증이다.

B에서 과적합이 완화되지만 validation 최고값이 낮아지면 backbone 비율0.3을 다음 후보로 둔다. A/B/C를 처음부터 하나로 합쳐 실행하면 warmup·LR·head 준비 중 어느 요인이 유효한지 알 수 없으므로 순서를 유지한다.

## 두 번째 실험: 큰 head의 용량 제한

현재 all-patch head의 입력 token은 유지하면서 hidden token 수만 바꾼다. H는 attention head 개수가 아니라 MLP hidden width를 정하는 값이다.

| Task | 현재 H | 후보 H | 현재 head parameter | 후보 head parameter | 기존 근거 |
|---|---|---|---|---|---|
| TUAB | 10 | 4 | 64,402,401 | 25,761,201 | 해당 TUAB 비교 미실행, 가설 |
| HMC | 30 | 4 | 145,207,205 | 19,362,005 | 과거 GR2-2 5-seed BAcc 개선 |

HMC의 과거 H4 비교는 validation BAcc0.7324→0.7353, test BAcc0.7457→0.7479였다. Weighted-F1·Kappa는 하락했지만 이번 사용자의 BAcc 우선 목표에는 재검증 가치가 있다. 현재 nearest weight에서 같은 개선을 보장하거나 warmup 변경 효과와 더해서 예측하지 않는다.
TUAB는 6,440만 parameter의 head를 2,576만으로 줄이는 후보이다. Dropout0.1→0.4는 과거 TUAB에서 불리했으므로 dropout 증가를 첫 처방으로 두지 않는다.

## 그 다음 방법과 task별 적용 순서

- MentalArithmetic: 우선 A/B/C로 후반 과적합을 줄인다. 현재5seed 중4개의 best가 epoch1–4이고 후반 기록 loss는 거의0이다. 과거 H2 head가 BAcc를 개선하지 못한 기록도 있어 단순 축소를 우선하지 않는다.
- TUAB: A/B/C 다음 H4를 한 요인으로 비교한다. AUROC도 하락하므로 threshold 조정만으로 해결되는 현상으로 보지 않는다.
- HMC: H4를 현재 weight에서 비교한다. 필요하면 train class count에 근거한 multiclass class weighting을 별도 loss arm으로 비교한다. Validation N1 recall이 약0.4로 낮지만 가중치는 validation/test 빈도가 아니라 train 빈도로만 정한다.
- SEED-V: 현재 차이는 약0.12 percentage point이며 best는 epoch36–43이다. TUAB와 같은 초반 붕괴로 묶지 않는다. H4를 유지하고 공통 안정화 방법의 검증 결과를 먼저 본다.
- TUEV·CHB-MIT·FACED·PhysioNet-MI·ISRUC: 현재 보고 평균을 넘는 결과를 유지할 수 있는지 검증한다. 특히 CHB-MIT는 공통 warmup0의 성능 하락을 확인할 대표 task다.
- Siena: 현재2seed만으로 완료된5seed 결과라고 집계하지 않는다.
- 필요할 때 L2-SP를 다음 공통 후보로 둔다. 일반 weight decay는0을 기준으로 수축하고, L2-SP는 pretrained weight를 기준으로 이동을 제약한다. Head를 먼저 학습하거나 backbone LR을 줄여도 후반 표현이 악화될 때 동기가 있다. 계수는 validation으로 선택해야 하며 현재 데이터에서 검증된 값은 없다. [원 논문](https://proceedings.mlr.press/v80/li18a.html)
- Binary threshold 조정은 기존 고정threshold 결과와 구분되는 평가 방법이다. 비교 모델에도 같은 validation-only 규칙을 적용하지 않은 결과를 기존 논문 숫자와 직접 동등한 SOTA로 제시하지 않는다.

## 성공 판정과 실행 범위

1. 동일5seed의 validation BAcc 평균을 우선한다. 각 seed의 test 최고 checkpoint를 골라 합치지 않는다.
2. TUAB는 전체 최고 validation BAcc와 epoch6–20 성능을 함께 보고, head 준비 때문에 best epoch 번호만 이동했는지 구분한다.
3. 방법을 고른 뒤 10개 task의 동일 checkpoint·동일 평가 규칙·5seed 결과를 완성한다. 공통 warmup은 하나만 채택하고 task별로 test에 맞춰 바꾸지 않는다.
4. 10개 모두 최고 보고 평균 초과를 달성했는지는 완성된 표로 판단한다. 평균 BAcc 개선이나 일부 task 개선만으로 목표를 달성했다고 부르지 않는다.

2026-09-15 후속 지시에 따라 TUAB A/B/C와 H4, 각 5 seeds의 실행 코드를 [ablation/tuab](../ablation/tuab/README.md)에 구현했다. 연결 복구 후 완료된 baseline source/config manifest를 검증하고, 복사한 source에만 policy를 연결했다. 서버 원본 `src`는 보존했다. 네 arm 모두 실제 TUAB train 데이터·전체 모델·CPU DDP·checkpoint 저장/복원 검증을 통과했고, **17:49:46 UTC에 job array 27478589 (0-19%10), 총 20개를 제출했다.** 17:51:34 UTC 현재 20개 모두 `PENDING (Priority)`다. 실행 원장은 `outputs/tuab_recovery_20260915/submission.json`이다.

SEED-V·MentalArithmetic·ISRUC·HMC의 과거 고성능 설정은 [별도 대조 문서](FOUR_TASK_SUCCESS_RECIPES_20260915.md)에 정리했다. 네 task의 과거 좋은 weight와 설정을 보존하며, 새 제출은 TUAB가 우선이다.

## Audit

- 현재 결과 및 TUAB 전 epoch numeric 지표: `outputs/audits/downstream_hp_review_20260915/ten_task_followup.json`.
- 집계: `ten_task_strategy_summary.json`. 각 원본 result/validation 파일의 원격 경로를 포함한다.
- 이전 같은-weight HP 비교: [Downstream HP review](DOWNSTREAM_HPARAM_REVIEW_20260915.md). 이 문서의 task별 warmup 권장안은 이번 공통 warmup 비교 방향으로 대체한다.
- 재현: `E:/workspace/.venv-ablation/Scripts/python.exe outputs/audits/downstream_hp_review_20260915/ten_task_strategy.py`.
