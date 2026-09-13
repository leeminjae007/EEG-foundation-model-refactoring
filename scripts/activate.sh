# source scripts/activate.sh
EEG_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$EEG_PROJECT_ROOT/.venv/bin/activate"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$EEG_PROJECT_ROOT/cache/xdg"
export MPLCONFIGDIR="$EEG_PROJECT_ROOT/cache/matplotlib"
export NUMBA_CACHE_DIR="$EEG_PROJECT_ROOT/cache/numba"
export TORCH_HOME="$EEG_PROJECT_ROOT/cache/torch"
export PIP_CACHE_DIR="$EEG_PROJECT_ROOT/cache/pip"
export MNE_DONTWRITE_HOME=true
export TMPDIR="$EEG_PROJECT_ROOT/tmp"
export WANDB_DIR="$EEG_PROJECT_ROOT/outputs/wandb"
export WANDB_CACHE_DIR="$EEG_PROJECT_ROOT/cache/wandb"
export WANDB_DATA_DIR="$EEG_PROJECT_ROOT/cache/wandb-data"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PHYSIONET_MI_OUTPUT="$EEG_PROJECT_ROOT/processed/physionet-mi/processed_average"
