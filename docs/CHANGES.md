# 새 프로젝트에만 적용한 변경

2026-09-12 기본 마스킹 변경: 새 기본값은 GR9-1 모델 + geometry 50%, REVE 반경 3cm, 시간 2–15초입니다. 아래 GR9-1 기록은 기존 `configs/gr9_1.yaml` 재현 기준이며, 변경 내용과 새 검증은 [GEOMETRY_DEFAULT.md](GEOMETRY_DEFAULT.md)를 참조하세요.

원래 코드·환경·설정·checkpoint·로그를 덮어쓰거나 삭제하지 않았습니다. 기존 Slurm 학습은 중지·requeue·release하지 않았습니다. 이번 Slurm 작업은 새 프로젝트의 환경 설치와 소량 CPU 검증뿐입니다. 작업 ID는 `outputs/verification_jobs.json`에 있습니다.

## 단순화

- GR9-1의 patch tokenizer, Dual-3 encoder, additive concat PE, compact decoder만 유지했습니다.
- CBraMod tokenizer, historical D192/LayerNorm 초기화, full-add/concat-mixer fusion, 공간·시간 attention-logit bias, absolute-PE bias, component RMS/learned scaling을 제거했습니다. GR9-1에서 활성화되지 않는 경로입니다.
- Random/geometry/Leiden masking, variable-count decoder, branch auxiliary loss, frequency/phase loss를 제거했습니다. GR9-1의 I-JEPA target block 순서·overlap은 유지했습니다.
- Frozen/random transfer, LOSO 제어, early stopping, threshold 탐색, 실험 controller·자동 제출, W&B 온라인 업로드를 새 학습 코드에서 제거했습니다. 학습 log·진단·checkpoint는 새 폴더에 저장합니다.
- 별도 fusion.py, dual_path.py, 공용 utils.py, MLP wrapper는 만들지 않았습니다. Dataset registry와 subject_cv는 **보존 대상 데이터셋 코드**이므로 그대로 두었습니다.
- 모델 내부의 구형 checkpoint 호환 분기를 제거했습니다. 키 변경은 `scripts/convert_checkpoint.py`에서 명시적으로 처리합니다. Strict loading 검증 없이 이름만 바꾸어 저장하지 않습니다.

## Dataset·전처리 복사

기존 두 폴더의 33개 파일을 가져왔습니다. 5개는 byte-identical, 28개는 import 또는 파일 경로만 바뀌었습니다. 각 파일의 원본·복사본 SHA256과 unified diff는 [전체 복사 manifest](data_copy_manifest.json)에 있습니다. 함수·클래스·주석·예외처리·처리 로직은 보존했습니다.

경로 변경은 다음에 한정됩니다.

- `src.datasets` → `src.data.datasets`.
- `src.utils.electrode_geometry` → `src.data.electrode_geometry`. 필수 전극 좌표 보조 코드는 원본 그대로 복사했습니다.
- 기존 전처리 결과 출력 root → 새 프로젝트의 `processed/`.
- SEED-VIG의 home fileset 출력 root도 새 `processed/SEED-VIG/`로 변경했습니다.

실제 Siena 원시 처리는 기존 `src/preprocessing/preprocessing_siena.py`가 아니라 별도 pinned upstream 함수로 실행됐습니다. 따라서 `siena_official/`에 audit root의 원본 data_process.py, json_generate.py, siena_dataset.py 및 checksum 검증 기록도 추가했습니다. `run_siena_official.py`는 기존 wrapper의 import/리소스/출력 경로만 새 구조에 맞췄습니다. 새 변환 결과와 실행 ledger는 각각 `processed/siena/processed/`, `outputs/siena_preprocessing/`입니다. Upstream의 원래 예외처리와 source checksum 검사는 유지했습니다.

필수 보조 코드와 Siena resource의 개별 출처·hash·diff는 [보조 파일 manifest](data_support_manifest.json)에 있습니다.

Raw data와 현재 사용 중인 processed data 경로는 loader의 읽기 경로로 남겼습니다. Raw 전처리는 task별 원본 소스에 정의된 동작을 유지하며, 이번 작업에서 전체 원시 데이터 변환은 실행하지 않았습니다. ISRUC의 `edf_` dependency와 일부 원시 데이터 경로가 접근되지 않는 문제는 별도로 기록하고 코드를 임의로 고치지 않았습니다.

## 실행 중 확인한 문제와 처리

- 로그인 노드의 긴 설치/복사 process가 exit 137로 종료되어 CPU Slurm 작업에서 설치했습니다. 프로그램에서 예외를 삼키거나 재시도 fallback을 추가하지 않았습니다.
- 원래 torch 패키지가 공용 설치를 가리키는 symlink였습니다. 이를 복사한 뒤 `libmkl_intel_lp64.so` 미발견 오류가 실제 발생했습니다. 설치기가 **실제 symlink 대상**의 native library를 복사하도록 원인을 수정했습니다. 기존 환경에 패키지를 설치하지 않았습니다.
- Slurm이 sbatch script를 spool 경로로 복사하므로 `$0`에서 프로젝트 root를 찾는 방식이 실패했습니다. CPU 검증 script는 프로젝트 root에서 실행하고 제출 시 `--chdir`로 지정하도록 수정했습니다.
- 새 model/training 코드에 try/except, hasattr/getattr fallback은 추가하지 않았습니다. Model의 epsilon·빈 attention 행 처리·mask와 테스트 assert는 수학/검증의 일부로 유지했습니다. 보존 대상 데이터 처리 코드의 기존 예외처리는 변경하지 않았습니다.

구현 버그를 의심해 모델 수식을 바꾼 항목은 없습니다. “Binary CE”가 실제 weighted BCE인 점, I-JEPA target 블록 overlap, 각 블록별 target 반복은 원본 동작으로 명시하고 그대로 유지했습니다.
