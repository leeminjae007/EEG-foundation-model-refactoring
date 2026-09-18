#!/usr/bin/env bash
#SBATCH --account=system
#SBATCH --partition=gl40s_dev,gl40s_short,gl40s_long
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=outputs/knn37_local_search_20260916_1157/knnhp5-%x-%A_%a.out
#SBATCH --error=outputs/knn37_local_search_20260916_1157/knnhp5-%x-%A_%a.err
set -euo pipefail
cd /gpfs/data/oermannlab/users/ml10266/workspace/EEG-founation-model
action="${1:?train or evaluate}"
slug="${2:?dataset slug}"
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR="/tmp/knnhp5-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
exec .venv/bin/python -m scripts.run_knnhp_locked_five_seed "$action" \
  --slug "$slug" --index "$SLURM_ARRAY_TASK_ID"
