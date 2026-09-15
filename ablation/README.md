# Encoder / Positional Embedding ablation

기존 `src/`, `pretrain.py`, `finetune.py`, `configs/`는 수정하지 않았다. 이 디렉터리의 진입점이 실행되는 동안에만 기존 학습 엔진의 모델 생성 함수를 연결한다. 데이터 로더, loss, optimizer, scheduler, 평가 및 checkpoint 저장 코드를 재사용한다.

## 비교군

### Encoder comparison

| 설정 파일 (`ablation/configs/`) | 사용한 인코더 | 인코더 파라미터 | 전체 사전학습 파라미터 |
| --- | --- | ---: | ---: |
| `encoder_labram.yaml` | LaBraM base, 12 blocks, 10 heads, width 200 | 5,795,160 | 6,620,790 |
| `encoder_cbramod.yaml` | CBraMod 원본 criss-cross encoder, 12 layers | 4,831,200 | 5,656,830 |
| `encoder_csbrain.yaml` | CSBrain 원본 temporal/brain embedding + 12 layers | 8,809,200 | 9,634,830 |
| `encoder_mjde.yaml` | 현재 MJDE, 3 stages / 12 blocks | 5,787,200 | 6,612,830 |
| `encoder_mjde_lite.yaml` | 현재 MJDE의 첫 stage / 4 blocks | 1,929,200 | 2,754,830 |

공통 tokenizer, SHPE, decoder, reconstruction loss를 고정하고 **인코더 부분**을 교체한다. 원 논문의 별도 tokenizer, pretraining objective, 분류 head, 공개 pretrained weights는 이 비교에 포함하지 않는다. LaBraM의 CLS token과 최종 normalization은 인코더 구성으로 유지한다. LaBraM은 원본 constructor에 `qkv_bias=True`, `init_values=0.1`, q/k 및 block LayerNorm epsilon `1e-6`을 지정한다. 모델 간 파라미터 수를 강제로 맞추지는 않았다. MJDE-lite는 사용자와 정한 **1-stage** 변형이다.

**마스크 조건:** 원본 CBraMod/CSBrain은 가려진 위치를 포함한 전체 격자를 처리한다. 이 동작을 보존하기 위해 encoder 비교 5개 모두 `mask_mode: dense_zero`를 사용한다. 가려진 token+PE를 입구에서 0으로 만들고 모든 위치를 attention에 참여시킨 뒤, 출력에서 가려진 위치를 다시 0으로 만든다. 빈 위치는 내부에서 이웃 정보를 받을 수 있지만 가려진 정답 신호는 입력되지 않는다. 따라서 `encoder_mjde`는 기존 context-only MJDE 실행과 구별되는 공통 마스크 비교군이다.

### Positional embedding comparison

| 설정 파일 | 위치 임베딩 정의 |
| --- | --- |
| `pe_none.yaml` | encoder와 decoder 모두 PE 없음 |
| `pe_channel_id.yaml` | 고정된 채널 이름 ID의 학습 embedding + 기존 temporal SinCos/GELU/RMSNorm |
| `pe_acpe.yaml` | CBraMod 원본 `PatchEmbedding.positional_encoding`: depthwise Conv2d, kernel `(19, 7)` |
| `pe_reve4d.yaml` | REVE 원본 `FourierEmb4D` + `mlp_pos_embedding` + LayerNorm |
| `pe_shpe.yaml` | 현재 `src/modules/position_embedding.py`의 SHPE 그대로 |

PE 비교는 현재 MJDE 3-stage와 `mask_mode: context_only`를 고정한다. 기본 `pe_scope: both`는 encoder와 decoder의 PE를 함께 바꾼다. 인코더 PE만 비교하려면 별도 overlay에서 `pe_scope: encoder`로 설정하고 **비교군 전체에 동일하게** 적용한다. 이 경우 `none`도 decoder SHPE는 남아 있다.

- `channel_id`는 별도 논문 구현으로 주장하지 않는 기본 비교군이다. SHPE의 공간 부분을 채널 이름 lookup으로 바꾸고 temporal 부분과 후처리를 유지한다. vocabulary는 최초 config 해석 때 전체 지원 데이터셋과 MNE standard_1005로 만들어 checkpoint config에 저장한다. 새 데이터셋에서도 같은 채널은 같은 ID를 사용한다. 학습에서 보지 못한 채널 ID의 embedding은 초기값에서 시작한다.
- ACPE 전에 숨겨진 토큰을 제거한다. decoder ACPE에는 context projection만 전달하고 숨겨진 위치를 0으로 둔다. 정답 파형은 전달하지 않는다. Conv2d 원본 자체는 수정하지 않았다.
- REVE는 이 프로젝트의 정규화 좌표를 MNE standard_1020의 원래 미터 단위로 되돌린다. Fourier의 `freqs=4`, `increment_time=0.1`, `margin=0.4` 및 원본 차원 절단을 유지한다. encoder/decoder 폭은 공통 모델에 맞춰 200/100이다. PE 자체를 비교하도록 좌표 noise는 기본 `0.0`이다. 원본 모델의 좌표 augmentation도 추가하려면 별도 실험에서 `reve_noise_ratio: 0.0025`를 지정한다.
- `pe_shpe`는 기존 모델과 **동일한 seed에서 출력과 gradient가 bit-exact**함을 검사했다.

## 원본 코드와 연결 범위

원본 출처, 고정 커밋 및 파일별 SHA256은 [vendor/manifest.json](vendor/manifest.json)에 있다. `.gitattributes`로 원본 파일의 줄바꿈 변환도 막았다.

| 출처 | 고정 revision | 사용 부분 |
| --- | --- | --- |
| [LaBraM](https://github.com/935963004/LaBraM) | `c431221e6cfd23dbfa9950e0180682fb322b0548` | `NeuralTransformer` 원본 forward/blocks/norm/초기화 |
| [CBraMod](https://github.com/wjq-learning/CBraMod) | `b9e961003214326972c567eff390e75b0287e32a` | 원본 encoder 및 ACPE |
| [CSBrain](https://github.com/yuchen2199/CSBrain) | `185aee55b24d0410a830df8dd08d03f675616998` | 원본 모델/transformer, 데이터셋별 region/topology metadata |
| [REVE](https://github.com/elouayas/reve_eeg) | `06a7059a07c3dabd80aee60c3dbc1eca4bdbe1c7` | 원본 Fourier 4D 및 위치 MLP |
| 현재 MJDE/SHPE | 작업 시작 시 저장소 `a7c17a3f03d0239eec189522ec282c2273cfec34` | 기존 클래스를 직접 import |

원본 파일은 byte 그대로 보관한다. 원본의 `models.*` import만 해당 저장소 내부로 연결하며, 프로젝트 전체의 `sys.path`나 일반 `models` 모듈을 덮어쓰지 않는다. REVE는 전체 `encoder.py`를 보관하되 PE 클래스/함수 두 개의 원문만 추출해 실행한다. 함수 본문을 재작성하지 않으며, PE와 무관한 transformers/flash-attn/새 PyTorch 의존성을 불러오지 않는다.

연결 코드의 역할:

- [`encoders/`](encoders/): 원본 모델의 tokenizer를 token 입력 어댑터로, reconstruction projection을 Identity로 교체. 원본 encoder forward를 호출하고 `[B,C,T,D]`로 돌려준다. LaBraM 내부 PE는 공통 PE와 중복되지 않도록 제거한다.
- [`positions/`](positions/): PE 크기·단위·호출 형태를 연결한다.
- [`models.py`](models.py): 기존 tokenizer/decoder 모듈을 재사용한다. 기존 초기화가 끝난 뒤 논문 모듈을 부착하므로 원본 초기화를 기존 Kaiming 함수가 덮지 않는다.
- [`integration.py`](integration.py): 실행 범위 안에서 기존 엔진의 모델 생성 함수 세 곳을 교체하고 종료 시 복원한다.
- [`bootstrap.py`](bootstrap.py): `src/data`가 없는 체크아웃에서는 **이미 추적 중인** `tests/reference/src` 데이터 소스에 namespace alias를 연결한다. `src/data`가 있는 SSH 서버에서는 그것을 사용한다. 선택된 경로는 실행 기록에 남긴다.

### CSBrain montage 연결에서 필요한 보완

원본 CSBrain 데이터셋 파일에서 region ID와 정렬 순서를 읽는다. 입력을 그 순서로 정렬하고, 출력은 기존 head가 기대하는 입력 채널 순서로 복구한다. fine-tuning 시 채널 수가 달라져도 region mask를 다시 구성하며, 해당 montage에 없는 region의 convolution은 gradient 대상에서 제외한다.

1. 원본 PhysioNet 파일은 `FT7/FT8`의 region을 2로 지정하지만 `topology[2]`에 두 채널을 빠뜨렸다. [`montage.py`](montage.py)에서 region 2의 앞에 원래 채널 목록 순서대로 추가한다. **원본 파일을 고친 것은 아니며, 이 한 건의 metadata 보완을 명시적으로 적용한다.**
2. MentalArithmetic의 마지막 `A2-A1`은 기존 데이터셋에서 제외되는 reference 채널이다. 원본 CSBrain stress wrapper와 같이 이를 제외해 인코딩하고, 공통 출력 격자에는 0으로 돌려놓는다.

CBraMod/LaBraM/REVE의 제공 LICENSE를 원본과 함께 보관했다. 이 CSBrain revision에는 LICENSE 파일이 없어 README와 출처를 보관했다.

## SSH 터미널 실행

아래 작업은 저장소가 있는 서버에서 실행한다. 실제 전체 학습은 GPU가 할당된 노드에서 실행한다.

### 1. 환경과 검사

```bash
cd /gpfs/data/oermannlab/users/ml10266/workspace/EEG-founation-model
source scripts/activate.sh

# 기존 Python 3.8 / torch 2.0.1 환경에 추가 의존성만 설치
python -m pip install -r ablation/requirements.txt

# 원본 파일 해시 검사: 인터넷 접속 없이 실행
python -m ablation.sources

# 데이터·GPU 없이 10개 설정의 forward/backward/optimizer/복원/누출 검사
python -m ablation.smoke

# 연결 및 원본 동등성 검사. LMDB 검사용 임시 디렉터리는 전용 경로 사용.
python -m pytest ablation/tests -q --basetemp=tmp/ablation-tests

# 데이터 접근 없이 최종 설정과 파라미터 수 확인
python -m ablation.pretrain \
  --config ablation/configs/encoder_labram.yaml --dry-run
```

서버에 기존 `.venv`가 없다면 프로젝트의 `scripts/install_environment.py`/기존 환경 설치 안내를 먼저 따른다. `ablation/requirements.txt`는 프로젝트 전체 의존성 설치 파일이 아니다.

### 2. 한 비교군 사전학습

```bash
# 4 GPUs, 기존 기본값인 GPU당 batch 128 / global batch 512
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  --module ablation.pretrain \
  --config ablation/configs/encoder_labram.yaml --distributed

# PE 비교: 설정 파일만 변경
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  --module ablation.pretrain \
  --config ablation/configs/pe_acpe.yaml --distributed
```

`--data-dir /실제/TUEG_LMDB`로 데이터 경로를 바꿀 수 있다. `--seed 1234`, `--batch-size 64`(GPU당), `--output outputs/ablation/custom_run`도 지원한다. 비교에서는 GPU 수, effective batch size, 학습 길이와 seed 조건을 맞춘다. 메모리 사용량은 모델마다 다르며 위 batch가 모든 GPU에서 들어간다는 의미는 아니다.

기본 checkpoint 경로는 `outputs/ablation/<설정의 name>/seed<seed>/last.pth`다. 기존 run이 있으면 `--resume` 또는 새 `--output`을 요구한다.

```bash
# 같은 설정으로 학습 재개
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  --module ablation.pretrain \
  --config ablation/configs/encoder_labram.yaml --distributed \
  --resume outputs/ablation/encoder_labram/seed42/last.pth

# 실제 TUEG 데이터 두 샘플로 한 update만 검사
python -m ablation.pretrain --config ablation/configs/encoder_labram.yaml \
  --device cpu --smoke
```

`--smoke`는 실제 데이터가 필요하고 `outputs/smoke/encoder_labram_seed42/`에 기록한다. 합성 검사는 별도 `ablation.smoke` 명령이다. smoke checkpoint는 전체 학습 재개나 downstream 비교에 사용할 수 없게 했다. 재개할 때는 원래 데이터·seed·optimization·모델 설정을 유지한다.

### 3. Downstream fine-tuning

```bash
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  --module ablation.finetune --distributed \
  --config configs/downstream/gr9-1_warmup5_seedv_seed42.yaml \
  --checkpoint outputs/ablation/encoder_labram/seed42/last.pth
```

다른 task/seed는 기존 `configs/downstream/` 파일을 지정한다. architecture와 PE는 **pretraining checkpoint의 config로 복원**하고 가중치는 strict loading한다. 과거 MJDE checkpoint를 다른 인코더로 부분 loading하거나 shape가 다른 가중치를 무시하지 않는다. `--dry-run`은 checkpoint·montage·head 구성을 확인하고 데이터는 읽지 않는다. fine-tuning 재개는 같은 명령에 `--resume <해당 downstream run>/last.pth`를 추가한다.

기본 downstream 출력은 `outputs/ablation/<name>/finetune/<dataset>/<downstream-config-stem>_seed<seed>/`다. LR이 다른 기존 config도 별도 run에 저장한다. 필요하면 `--output`으로 짧은 경로를 지정한다. 결과 파일은 기존 엔진과 동일한 `result.json`, `validation.jsonl`, `metrics.jsonl`, `best-*.pth`, `last.pth`다.

### 4. 전체 그룹 순차 실행 / SSH 접속 종료 후 유지

```bash
# 할당된 노드에서; 5개 모델을 동시에 올리지 않고 차례로 학습
bash ablation/scripts/pretrain_sweep.sh encoder 4 /실제/TUEG_LMDB 42
bash ablation/scripts/pretrain_sweep.sh pe      4 /실제/TUEG_LMDB 42
# 두 그룹 모두: 첫 인자를 all로 지정
```

Slurm에서는 다음 명령으로 제출한다. `a100_short,a100_long` 중 가능한 파티션에서 **총 A100 4개**를 1~4개 노드에 배정받는다. `1노드×4GPU`, `2노드×2GPU`, `4노드×1GPU`, `2+1+1` 배치 모두 같은 world size 4 / global batch 512로 실행한다. GPU마다 CPU 8개와 RAM 32 GB, 제한 시간 24시간을 요청한다. 실제 시작 시각은 클러스터 자원과 계정/QOS 제한에 따라 정해진다.

```bash
git pull --ff-only origin main
source scripts/activate.sh
python -m pip install -r ablation/requirements.txt

# 인코더 5개를 각각 별도 job으로 제출
bash ablation/scripts/run_pretrain.sh encoder
# REVE PE만 제출
bash ablation/scripts/run_reve.sh
# 개별 arm / seed / 학습 추가 인자
bash ablation/scripts/run_pretrain.sh encoder_labram 42
bash ablation/scripts/run_pretrain.sh encoder_labram 42 \
  --resume outputs/ablation/encoder_labram/seed42/last.pth

# 예전 1노드 고정 요청으로 대기 중이라면, 해당 그룹의 PENDING job만 교체
bash ablation/scripts/run_pretrain.sh --replace-pending encoder
bash ablation/scripts/run_reve.sh --replace-pending
squeue -u "$USER"
```

제출기는 같은 사용자·job 이름이 이미 대기/실행 중이면 건너뛴다. `--replace-pending`은 선택한 arm과 seed의 대기 job만 취소하고 새 설정으로 제출하며, 실행 중인 job은 유지한다. 새 스크립트를 pull해도 기존 job의 자원 요청은 바뀌지 않는다. 완료한 run의 checkpoint가 있으면 `--resume` 또는 별도 `--output`을 지정한다.

[`scripts/pretrain_flexible.slurm`](scripts/pretrain_flexible.slurm)은 `--nodes=1-4 --ntasks=4 --gpus-per-task=a100:1`과 `srun`을 사용한다. 각 프로세스는 Slurm이 지정한 GPU 하나만 보고, [`scripts/slurm_worker.sh`](scripts/slurm_worker.sh)가 기존 엔진에 `RANK`, `WORLD_SIZE`, `LOCAL_RANK=0`을 연결한다. 노드 로컬 task 번호를 CUDA 번호로 쓰지 않는다. 첫 노드의 주소와 공통 포트에서 분산 학습을 초기화한다. 포트 충돌 시 `ABLATION_MASTER_PORT`를 지정할 수 있다. [Slurm 자원 요청 문서](https://slurm.schedmd.com/sbatch.html), [GPU 바인딩 문서](https://slurm.schedmd.com/srun.html).

이전 [`scripts/pretrain.slurm`](scripts/pretrain.slurm)은 기존 명령의 호환성을 위해 1노드 전용으로 유지한다. 여러 노드 실행에는 위의 새 제출기를 사용한다.

제출은 이 구현 작업에서 실행하지 않았다. Slurm job은 SSH 연결을 종료해도 계속 실행된다. Slurm이 없는 전용 GPU 서버라면 `tmux new -s eeg-ablation` 안에서 위 학습 명령을 실행하고 `Ctrl-b`, `d`로 빠져나온다.

## 검증 범위

- 10개 preset 모두 원본 깊이를 유지한 합성 forward/backward, 모든 trainable parameter의 gradient, optimizer update, 숨겨진 파형 변경에 대한 출력 불변성, strict checkpoint roundtrip을 통과했다.
- 기존 MJDE+SHPE 출력·gradient 동등성, 공통 tokenizer/decoder 초기화 동등성을 검사했다.
- 14개 downstream montage에서 CSBrain 연결을 검사했다. 10개 arm의 pretrain → SEED-V strict loading/gradient/optimizer 그룹, ISRUC sequence head도 검사했다.
- 합성 LMDB로 **기존 학습 엔진**의 사전학습·중단 후 재개·downstream 학습·validation 선택·test 평가를 실행했다. 재개와 연속 실행의 최종 가중치가 정확히 같다.
- Slurm 실행기 검사 12개에서 네 가지 GPU 배치의 rank 연결, REVE/encoder 제출, 학습 인자 전달, 대기 job 교체와 실행 중인 job 보존을 검사했다. 로컬 CPU 프로세스 4개를 실제 Gloo/DDP로 연결해 gradient 평균과 동일한 optimizer update도 확인했다. 실제 Slurm 스케줄링이나 노드 간 NCCL 통신을 검증한 것은 아니다.
- 로컬 검증 환경은 Windows / Python 3.12 / PyTorch 2.5.1 CPU다. 새 Python 코드의 Python 3.8 문법과 SSH용 dependency의 Python 3.8 지원, shell 문법을 별도로 확인했다. 실제 서버의 torch 2.0.1 CUDA/BF16/DDP, 실데이터 전체 학습과 성능 수치는 아직 실행하지 않았다.

합성 검사 수치는 `outputs/ablation/synthetic_checks.json`에 저장된다. 이는 성능 비교 결과가 아니다.
