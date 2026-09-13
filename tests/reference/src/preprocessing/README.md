# Offline processed-data builders

이 디렉터리의 스크립트는 CBraMod에서 가져온 dataset별 **offline**
preprocessing reference다. `src/downstream.py`와 dataset loader는 이
스크립트를 import하거나 실행하지 않으며, 이미 만들어진 데이터를 직접
읽는다.

확인된 출력 계약:

| Dataset | Output | Stored sample |
|---|---|---|
| BCI-IV-2a | split-indexed LMDB | `[22,4,200]`, class `0..3` |
| FACED | split-indexed LMDB | `[32,10,200]`, class `0..8` |
| Mumtaz | split-indexed LMDB | `[19,5,200]`, binary |
| PhysioNet-MI | split-indexed LMDB | `[64,4,200]`, class `0..3` |
| SEED-V | split-indexed LMDB | `[62,1,200]`, class `0..4` |
| SEED-VIG | split-indexed LMDB | `[17,8,200]`, scalar |
| BCIC2020-3 | split-indexed LMDB | `[64,3,200]`, class `0..4` |
| MentalArithmetic | split-indexed LMDB | `[20,5,200]`, binary |
| TUAB | split pickle folders | `{"X": [16,2000], "y": 0|1}` |
| TUEV | split pickle folders | `{"signal": [16,1000], "label": [1..6]}` |
| HMC | subject-disjoint split pickle folders | `{"X": [4,6000], "ch_names": [...], "y": 0..4}` |
| TUEG | LMDB | `[19,30,200]` |

LMDB downstream 데이터는 `__keys__`에 `train`, `val`, `test` key 목록을
저장하고 각 sample은 `{"sample": array, "label": value}` 형식이다.

스크립트의 raw input/output 경로는 과거 데이터 배치를 기록한 provenance이며
현재 단일 data root의 loader 경로가 아니다. 새 데이터 준비에 사용할 때는
대상 데이터의 실제 경로와 channel order를 먼저 확정해야 한다. FACED는 공개된
30-channel acquisition order와 뒤따르는 A2/A1 mastoid order를 loader에
명시했고 mastoid는 active EEG에서 제외한다. SEED-VIG도 원 17-channel order를
명시했다. Speech의 기존 processed record에는 channel-name metadata가
없으므로 임의 좌표를 부여하지 않으며, metadata가 확정되기 전에는 loader가
fail-fast한다.

HMC는 정렬된 151개 recording을 `100/25/26`명의
`train/val/test`로 분리한다. 네 EEG derivation(F4-M1, C4-M1, O2-M1,
C3-M2)을 0.1--75 Hz band-pass 및 50 Hz notch filter한 뒤 200 Hz로
resample하고, 공식 annotation의 유효한 30초 W/N1/N2/N3/R epoch만
저장한다. 실행 전 공식 `SHA256SUMS.txt` 전체 검증을 통과해야 한다.
