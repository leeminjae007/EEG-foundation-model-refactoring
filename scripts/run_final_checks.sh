#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export XDG_CACHE_HOME="$PWD/cache/xdg"
export MPLCONFIGDIR="$PWD/cache/matplotlib"
export MNE_DONTWRITE_HOME=true
export NUMBA_CACHE_DIR="$PWD/cache/numba"
export TMPDIR="$PWD/tmp"
.venv/bin/python -m pytest tests/test_reference_history.py tests/test_engine.py -q
.venv/bin/python -m pip check
.venv/bin/python -m compileall -q src pretrain.py finetune.py scripts/convert_checkpoint.py
