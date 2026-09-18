# EEG-founation model

GR9-1 모델을 읽기 쉽게 분리한 연구 코드입니다. 기본 사전학습은 50% geometry tubelet 마스킹을 사용합니다. 원래 프로젝트와 독립된 `EEG-founation-model` 폴더에서 실행합니다.

사전학습 흐름은 **EEG → 시간·주파수 토큰 → 위치 정보 더하기 → context만 인코딩 → mask token과 위치 정보로 target 복원 → Smooth L1**입니다. Mask는 먼저 뽑지만, token을 가리는 시점은 원본처럼 위치 정보를 더한 뒤입니다.

- [src/model.py](src/model.py): 위 흐름, 일반 downstream 모델, ISRUC sequence 모델.
- [src/encoder.py](src/encoder.py): 세 단계의 `공간→시간`, `시간→공간`, feature별 gate 합산. Norm·residual·MLP 순서를 직접 보여 줍니다.
- [src/decoder.py](src/decoder.py): target 블록마다 `[context, mask+PE]`를 만들고 네 decoder block으로 복원합니다.
- `src/modules/`: tokenizer, attention, SH·시간 PE, mask 인덱스, RMSNorm, loss 수식.
- `src/data/`: 복사한 dataset·전처리 코드와 필수 전극 좌표 코드. 처리 로직은 보존했습니다.
- `src/training/`: 학습 루프, 평가, 진단, LR schedule, checkpoint 저장·복원.

MLP는 사용하는 block 안의 `nn.Sequential`로 충분하므로 별도 wrapper 파일을 만들지 않았습니다.

## 환경

이 작업 공간의 `.venv`는 이미 구성했습니다. Python 3.8.16, 실제 GR9-1과 같은 PyTorch 2.0.1 / CUDA 11.8 binary를 사용합니다. 기존 환경의 torch symlink를 따라가지 않고 Python 패키지와 필요한 native library를 새 환경에 복사했습니다. 사용자 site-packages를 사용하지 않습니다.

```bash
cd /gpfs/data/oermannlab/users/ml10266/workspace/EEG-founation-model
source scripts/activate.sh
```

재설치는 다음과 같습니다. 설치는 충분한 메모리가 할당된 CPU 작업에서 실행하세요.

```bash
/gpfs/data/oermannlab/users/ml10266/.conda/envs/eegfm/bin/python -B \
  scripts/install_environment.py \
  --source-env /gpfs/data/oermannlab/users/ml10266/.conda/envs/eegfm
```

[requirements.txt](requirements.txt)는 필요한 직접 의존성, [requirements-lock.txt](requirements-lock.txt)는 설치된 전체 버전, [native library 기록](docs/torch_runtime_copy.json)은 실제 binary 출처입니다. Venv의 표준 라이브러리와 interpreter 기반은 Python 3.8.16 설치를 사용하지만, 학습 패키지와 프로젝트 소스는 독립적입니다.

## 실행

가중치 이름이 달라졌으므로 모델 밖에서 한 번 변환합니다. 이 작업 공간에는 변환된 epoch-40 가중치도 준비했습니다. 변환기는 누락·중복·shape 오류를 검사하고 strict loading으로 확인하며, 기존 destination을 덮어쓰지 않습니다.

```bash
python scripts/convert_checkpoint.py \
  tests/reference/checkpoint-epoch-0040.pth outputs/gr9_1_epoch40.pth
```

다음은 GPU를 **할당받은 후** 실행하는 명령입니다. 최종 geometry 설정의 전체 사전학습은 Slurm `27393348`로 제출했습니다. 아래 명령은 수동 실행 예시입니다.

```bash
# 4 GPU, GPU당 128개 → global batch 512. 40 epoch, warmup 없음.
python -m torch.distributed.run --standalone --nproc_per_node=4 pretrain.py \
  --config configs/pretrain.yaml --distributed

# 예: SEED-V, seed42. 다른 task/seed는 configs/downstream/의 파일을 지정.
python -m torch.distributed.run --standalone --nproc_per_node=1 finetune.py \
  --config configs/downstream/gr9-1_warmup5_seedv_seed42.yaml --distributed
```

기본 설정은 [configs/pretrain.yaml](configs/pretrain.yaml)입니다. REVE와 같은 실제 3D 거리 반경 3cm × 연속 2–15초 영역을 모아 570개 패치 중 285개(50%)를 target으로 사용합니다. 겹친 패치는 합집합에서 한 번만 포함하고, 나머지 285개는 모두 context입니다. 초기 선택 근거와 사용자 수정·제출 기록은 [geometry 비교 기록](docs/GEOMETRY_DEFAULT.md)에 있습니다. 현재 19채널에서는 3cm 이웃이 중심 전극 자신뿐이므로 한 전극의 시간 구간을 마스킹합니다.

기존 GR9-1의 I-JEPA 마스킹을 재현하려면 `--config configs/gr9_1.yaml`을 지정하세요. 변환된 `outputs/gr9_1_epoch40.pth`는 기존 I-JEPA 학습 가중치입니다. 새 geometry 조합의 학습 가중치나 성능으로 해석하면 안 됩니다.

Downstream 설정은 13 task × 5 seed와 추가 TUAB LR 5e-6 arm입니다. 2026-09-12 채택 정책대로 **모든 task에 warmup 5 epoch**를 사용합니다. GR9-1 pretrain 자체에는 warmup이 없습니다. 검증 BAcc와 AUROC/Kappa를 같은 run에서 선택하고, 학습 종료 후 각 선택 가중치로 test를 평가합니다.

```bash
# 실제 데이터의 두 샘플로 한 update만 실행. FP32 CPU 검증용.
python pretrain.py --device cpu --smoke
python finetune.py --config configs/downstream/gr9-1_warmup5_seedv_seed42.yaml --device cpu --smoke

# 새 프로젝트에서 저장한 epoch checkpoint의 학습 재개
python -m torch.distributed.run --standalone --nproc_per_node=4 pretrain.py \
  --distributed --resume outputs/pretrain_geometry50_reve3cm_t2_15/last.pth

# 고정 mask·동일 가중치·dropout RNG·gradient·dataset·checkpoint 검증
bash scripts/run_cpu_checks.sh
```

`--smoke` checkpoint는 일부 batch만 실행한 검증 산출물입니다. 전체 학습 재개에 사용하지 마세요. 구형 checkpoint 변환은 가중치 이식이며 구형 optimizer 상태의 재개 변환은 아닙니다.

모든 기본 출력은 새 폴더의 `outputs/`, cache는 `cache/`, 새 전처리 결과는 `processed/`입니다. 원본·기존 처리 데이터는 loader에서 읽기 전용으로 공유합니다. 전처리 실행도 위 환경 활성화 후 새 폴더에서 진행하세요. 자동 전처리나 전체 데이터 재생성은 실행하지 않습니다.

실제 기준과 한계는 [GR9-1 조사 기록](docs/GR9_1_BASELINE.md), 복사·삭제 내역은 [변경 기록](docs/CHANGES.md), 검증 수치와 미검증 항목은 [검증 보고서](docs/VALIDATION.md)에 있습니다.

Mumtaz·MentalArithmetic·BCIC-IV-2a의 과적합 조정 실험과 설정은 [fine-tuning 조정 기록](docs/FINETUNE_REGULARIZATION.md)에 있습니다. 첫 LR 후보 15개는 배열 `27393566`으로 제출했습니다.
