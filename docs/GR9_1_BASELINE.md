# 실제 GR9-1 기준

2026-09-12 기본 마스킹 변경: 새 기본값은 GR9-1 모델 + geometry 50%, REVE 반경 3cm, 시간 2–15초입니다. 아래 GR9-1 기록은 기존 `configs/gr9_1.yaml` 재현 기준이며, 변경 내용과 새 검증은 [GEOMETRY_DEFAULT.md](GEOMETRY_DEFAULT.md)를 참조하세요.

## 확인한 실행 산출물

- 완료된 pretrain: Slurm `27276586`, trial `gr9_1_equal_shpe_gelu_rmsnorm_d200_e40_20260910`, epoch 40.
- 기록된 명령: `/gpfs/data/oermannlab/users/ml10266/.conda/envs/eegfm/bin/python3.8 -m src.train --config /gpfs/data/oermannlab/users/ml10266/workspace/eeg-foundation-model/configs/gr9/pretrain_gr9-1_equal_pe_gelu_rmsnorm.yaml`. Slurm wrapper는 각 rank에 RANK/LOCAL_RANK/WORLD_SIZE를 지정했습니다.
- 실제 topology: `a100-[4017,4020,4022,4025]`, 네 노드에 각 A100 한 개, rank당 CPU 8개·host memory 32 GB. GPU당 batch 128, global batch 512.
- 근거: [실행 metadata](original_execution.json), [실제 resolved config](original_resolved_config.yaml), [실행 reproducibility](original_reproducibility.json), [checkpoint 경로·SHA256](checkpoint_source.json).
- Checkpoint에는 `context_encoder`, `decoder`, optimizer, scheduler, epoch, resolved config, RNG 상태가 있습니다. 모델 state 192개 tensor, 학습 parameter tensor 190개가 새 구현에 빠짐없이 대응합니다.

## 코드 버전의 한계

원래 디렉터리는 Git 저장소가 아닙니다. 실행 기록의 source-tree hash는 `4d82e9cf24289eeb203776f927fb9c82142ee227c501aa7a573b743a5b29a983`이지만 해당 hash에 대응하는 완전한 실행 당시 소스 아카이브는 확인하지 못했습니다. 따라서 현재 코드가 실행 당시 모든 줄과 같다고 주장하지 않습니다.

현재 소스를 새 폴더 `tests/reference/`에 복사해 비교 기준으로 고정했습니다. 또한 2026-09-12 follow-up 변경 전 아카이브를 `tests/reference_before/`에 보관했습니다. 두 소스 사이에는 GR9-5 attention-bias 경로와 진단 기능 등의 추가가 있으며, [변경된 파일과 hash](source_snapshot_changes.json)를 기록했습니다. GR9-1이 선택하는 실제 경로는 checkpoint strict loading 및 수치 replay로 확인합니다. 테스트 전용 원본에는 다른 variant도 들어 있지만, 배포용 `src/`에서는 사용하지 않습니다.

## 입력과 tokenizer

입력은 `[B,C,L]`, TUEG는 `[B,19,6000]`입니다. 200 Hz, 30초 신호를 겹치지 않는 1초·200 sample patch로 나눕니다. C→T 순서의 token grid는 `[B,19,30,200]`입니다.

시간 경로는 패치를 독립 sample로 보아 `[B*C*T,1,200]`에서 Conv1d→GroupNorm(8 groups)→GELU를 세 번 적용합니다. 채널 수 64/128/200, kernel 15/7/3, stride 5/3/2, padding은 kernel의 절반입니다. 결과 `[B*C*T,200,7]`를 펼쳐 1400→100 linear로 바꿉니다.

주파수 경로는 같은 patch에 `rfft(norm='forward')`→절댓값→`log1p`→Linear(101,100)→GELU→Linear(100,100)을 적용합니다. 시간 100과 주파수 100을 concat하고 dropout 0.1을 적용합니다. 모든 채널·시점에서 tokenizer 가중치를 공유합니다. CBraMod Conv2d tokenizer는 GR9-1에 사용되지 않습니다.

## 위치와 마스킹

MNE 1.6.1 `standard_1020` 좌표 해석은 복사한 `electrode_geometry.py`를 사용합니다. 좌표를 단위 구면으로 정규화하고 degree 0…4, 각 degree의 order -l…l 순서로 25개 real SH basis를 계산합니다. 좌표는 detach됩니다. SH는 encoder 25→100, decoder 25→50의 bias 없는 learned linear로 투영합니다.

시간 PE는 patch index 0…T-1에 대한 고정 SinCos입니다. 주파수는 `exp(-log(10000) * arange(half)/(half-1))`, 출력 순서는 sin 부분 뒤 cos 부분입니다. Encoder 시간 폭 100, decoder 시간 폭 50입니다. 공간·시간 concat 전체에 GELU→학습 가능한 RMSNorm(eps 1e-6)을 적용합니다. 별도 component RMS나 alpha/beta는 없습니다. Encoder와 decoder PE는 독립 parameter입니다.

I-JEPA는 batch마다 공통 target 크기와 context 크기를 뽑습니다. Target scale 0.15, aspect 0.75…1.5, target 4개입니다. Context scale 0.85…1.0, aspect는 C/T입니다. 직사각형은 채널 index와 시간 index의 격자이며 해부학적 거리로 뽑지 않습니다. Device-global torch RNG를 사용합니다.

Target 블록끼리는 겹칠 수 있습니다. `allow_overlap=false`는 context에서 target union을 제외한다는 뜻입니다. Context 후보를 최대 20회 뽑고 최소 10 token 조건을 만족하면 멈춥니다. Batch에서 가장 작은 context 개수에 맞춰 C→T 순서의 앞부분을 유지합니다. Context/target 외에 사용하지 않는 token도 있을 수 있습니다.

Raw tokenizer는 숨길 patch도 계산하지만 GroupNorm이 patch별이므로 visible patch와 통계를 공유하지 않습니다. Tokenizer→PE 더하기→channel validity 적용→encoder 입구에서 context 외 token을 0으로 만드는 순서를 유지합니다. Attention key와 query에서도 숨긴 위치를 제외합니다.

## Encoder

`depth=12`는 12개 독립 axial block을 뜻합니다. 3단계 × 단계마다 4개 block입니다. 각 단계에서 같은 fused 입력으로 S→T와 T→S 경로를 실행합니다. 경로 사이·단계 사이에 attention/MLP 가중치를 공유하지 않습니다.

한 axial block의 순서: mask 적용 → RMSNorm → attention → residual 더하기 → RMSNorm → Linear(200,800)·GELU·dropout·Linear(800,200)·dropout → residual 더하기 → mask 적용. Norm eps 1e-5이며 FP32로 RMS를 계산한 뒤 입력 dtype으로 돌립니다. Head는 4개입니다. Attention probability dropout은 0.1, encoder attention 출력에는 별도 projection dropout이 없습니다. DropPath는 0이므로 제거했습니다.

단계별 feature gate `[200]`의 sigmoid를 g라 할 때 `(g*S2T + (1-g)*T2S)*mask`입니다. Gate logit은 0으로 시작합니다. 최종 fused 출력에는 한 번의 output RMSNorm과 mask를 적용합니다. SH·시간 attention-logit bias는 GR9-1에 없습니다.

## Decoder와 loss

Encoder grid를 200→100 linear로 바꾼 뒤 **원래 C×T grid의 decoder PE**를 더합니다. 선택한 context를 C→T 순서로 모으고, 각 target 블록의 PE에는 zero-initialized learned mask token을 더합니다. Context를 4개 target 블록에 각각 반복하여 `[B*4, N_context+N_target,100]` sequence를 만듭니다.

네 decoder block은 전역 sequence attention이며 encoder의 axial block과 다릅니다. RMSNorm→attention→residual→RMSNorm→MLP→residual 순서, attention projection dropout도 0.1입니다. 최종 RMSNorm 뒤 target 구간만 잘라 100→200 reconstruction linear를 적용합니다. 결과는 `[B,4*N_target,200]`, 블록 순서와 각 블록의 C→T 순서를 유지합니다.

Target은 입력 EEG/100의 raw patch이며 no_grad에서 모읍니다. Target 정규화, frequency loss, phase loss는 없습니다. Smooth L1 beta 0.1을 scalar sample마다 계산하여 유효 target scalar sample 총수로 나눕니다. 겹친 target은 각 블록의 loss에 반복 포함됩니다. Loss 호출을 engine으로 분리해도 gradient 경로는 같습니다.

## 데이터와 학습

TUEG의 기존 LMDB를 readonly로 엽니다. 1,109,545개 sample, ordered-key SHA256 `12b507f7db2200124239d120a9214790cfb4ac2eab6cd55ecc62795a78cef29b`가 실행 당시 fingerprint와 같습니다. Dataset은 저장된 `[19,30,200]`을 reshape할 뿐이며 engine에서 100으로 나눕니다. 원시 TUEG 전처리 코드도 보존했지만 이번 작업에서 전체 전처리를 다시 실행하지 않았습니다.

Pretrain은 seed42, AdamW(모든 parameter에 WD 0.05), LR 5e-4, betas 0.9/0.999, eps 1e-8, 40 epoch, cosine minimum 1e-5입니다. Optimizer step 뒤 cosine step을 실행합니다. BF16 autocast, GradScaler 없음, clipping 1.0입니다. Accumulation 1이며 일반 accumulation에서도 불완전한 마지막 group은 제외합니다. DistributedSampler의 shuffle/drop_last와 worker seed 및 epoch별 loader generator seed를 유지합니다.

초기화는 원본처럼 전체 encoder·decoder 생성 후 Linear/Conv1d weight에 Kaiming normal fan_out/relu를 적용합니다. 생성 순서와 device 이동 시점, bias의 기본 초기화, norm weight=1, mask token=0, fusion gate=0을 보존했습니다. CPU replay에서 초기화 state 192개가 bit-exact입니다. CUDA의 난수는 CPU와 같다고 가정하지 않습니다.

## Downstream

설정 출처는 [75개 설정의 source manifest](config_manifest.json)입니다. 2026-09-12의 GR9-1 후속 정책을 적용했으며, pretrain 이후 새 warmup 정책이라는 점을 구분합니다. 기존 총 epoch, 배치, accumulation, task별 LR·WD·head 크기·dropout을 유지했습니다. 모든 task에 warmup 5 epoch, TUAB의 추가 LR 5e-6 arm도 포함합니다. Seed 순서는 42, 1234, 696, 1001, 3407입니다.

Roster: TUAB, TUEV, CHB-MIT, SEED-V, SEED-VIG, FACED, Mumtaz, MentalArithmetic(stress), BCIC-IV-2a, PhysioNet-MI(physio), BCIC2020-3(speech), ISRUC, HMC, Siena. 기존 loader의 split·channel 선택·resampling·scaling·sampling을 유지합니다. Coordinate-only 채널 활성화도 원본 downstream 진입점과 같습니다. Siena는 기존 공식 처리 결과와 29개 채널 loader를 사용하며 별도 정규화를 추가하지 않았습니다.

일반 head는 C→T→D 전체 feature flatten, Linear→GELU→dropout→Linear→GELU→dropout→Linear입니다. ISRUC만 20 sleep epoch의 projection(512)과 한 TransformerEncoder layer를 사용하는 별도 모델입니다. 모두 전체 fine-tuning입니다.

Binary weighted CE의 실제 구현은 **단일 logit의 BCEWithLogits**, training class count 역수의 mean-one weight를 label별 곱한 뒤 단순 평균입니다. Multiclass는 label smoothing을 포함한 CE, SEED-VIG는 MSE입니다. Binary threshold 0.5, AUPRC는 precision-recall 곡선의 사다리꼴 AUC이며 average_precision으로 바꾸지 않았습니다. Multiclass는 BAcc, Kappa, weighted F1, regression은 Pearson, R2, RMSE입니다.

Tokenizer·encoder(PE 포함)·head의 AdamW 세 LR group과 weight decay를 유지합니다. Downstream은 update **직전** GroupCosineScheduler를 진행하고 마지막 작은 accumulation group도 실제 group 크기로 나눕니다. Validation BAcc-best 및 AUROC/Kappa-best를 같은 학습에서 보관하고 종료 후 test를 평가합니다. Regression은 R2-best입니다. Early stopping이나 threshold tuning은 없습니다.
