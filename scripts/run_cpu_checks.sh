#!/usr/bin/env bash
set -euo pipefail
# 프로젝트 루트에서 실행한다. Slurm은 --chdir로 같은 위치를 지정한다.
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export XDG_CACHE_HOME="$PWD/cache/xdg"
export MPLCONFIGDIR="$PWD/cache/matplotlib"
export MNE_DONTWRITE_HOME=true
export NUMBA_CACHE_DIR="$PWD/cache/numba"
export TMPDIR="$PWD/tmp"
if [[ ! -f outputs/gr9_1_epoch40.pth ]]; then
    .venv/bin/python scripts/convert_checkpoint.py tests/reference/checkpoint-epoch-0040.pth outputs/gr9_1_epoch40.pth
fi
.venv/bin/python -m pytest tests/test_equivalence.py tests/test_downstream.py tests/test_schedule.py tests/test_geometry.py -q
.venv/bin/python pretrain.py --device cpu --smoke
.venv/bin/python finetune.py --config configs/downstream/gr9-1_seedv_seed42.yaml --device cpu --smoke
.venv/bin/python finetune.py --config configs/downstream/gr9-1_mentalarithmetic_seed42.yaml --device cpu --smoke
bash scripts/run_final_checks.sh
