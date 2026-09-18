# Four-task historical hyperparameter recipes

SEED-V, MentalArithmetic, ISRUC, HMC만 고정 5 seeds(42, 1234, 696, 1001, 3407)로 실행한다.
현재 nearest3–7 pretrained epoch40 weight에 과거 좋은 downstream HP를 적용한다.
표의 GR9-2/GR9-1/GR2-2는 HP를 확인한 과거 실험의 weight이며 이번 실행의 weight는 모두 현재 nearest3–7이다.

| Array index | Dataset | LR (tokenizer / encoder / head) | Warmup | Epochs | WD | Head dropout | Head | Batch × accumulation |
|---|---|---|---:|---:|---|---|---|---|
| 0–4 | SEED-V | 1e-4 / 1e-4 / 1e-4 | 0 | 50 | .01 | .1 | H4 | 64 × 1 |
| 5–9 | MentalArithmetic | 1e-4 / 1e-4 / 1e-4 | 0 | 50 | .01 | .2 | H5 (기본) | 64 × 1 |
| 10–14 | ISRUC | 1e-4 / 1e-4 / 1e-4 | 0 | 50 | .01 | .1 | 전용 sleep head | 2 × 32 |
| 15–19 | HMC | 1e-4 / 1e-4 / 1e-4 | 0 | 50 | .05 | .1 | H4 | 64 × 1 |

AdamW, cosine minimum LR 1e-6, clip1, BF16. Classification smoothing .1; binary MentalArithmetic에는 원래 구현대로 smoothing이 적용되지 않으며 class counts 1007/336의 weighted binary loss를 유지한다.
모든 epoch에서 validation BAcc-best를 선택하며, 같은 실행의 AUROC/Kappa selector도 보관한다. Early stopping은 없다.

완료된 nearest campaign source/config manifest를 검증하고 `src`를 byte-identical snapshot으로 복사한다.
현재 baseline 대비 변경은 네 task의 warmup5→0, HMC의 H30→4, 새 output 경로뿐이다.

## SSH 실행

새 `ablation/four_task/`가 서버에 복사된 뒤:

```bash
cd /gpfs/data/oermannlab/users/ml10266/workspace/EEG-founation-model
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m ablation.four_task.campaign all
```

Source/weight 검증 → 실제 train 데이터 4-task CPU DDP smoke → 20개 Slurm array 제출.
실행당 L40S 1 GPU, CPU8, RAM32GB, 4시간, 최대 동시10개. Partition은 gl40s_dev/short/long.
완료된 제출을 반복 실행하면 기존 job ID를 출력한다. 제출 응답이 불명확하면 원장을 보존하고 중복 제출을 막는다.

출력: `outputs/four_task_recipe_20260915/runs/<dataset>_seed<seed>/`.
제출 원장: `outputs/four_task_recipe_20260915/submission.json`.

## 제출 기록

- **Job array 27484155**, `0-19%10`, 2026-09-15 19:02:10 UTC 제출.
- 네 dataset 각각 5 seeds, 총 20개. 실제 train sample을 사용한 전체 모델 CPU DDP 2 updates와 checkpoint 저장·복원 검증 모두 통과했다.
- 원본 nearest baseline과 새 snapshot의 `src` 파일들이 byte-identical임을 확인했다.
- 19:03:37 UTC: 20개 모두 `PENDING (ReqNodeNotAvail, May be reserved for other job)`. 학습은 아직 시작하지 않았다.

```bash
squeue -r -j 27484155
```
