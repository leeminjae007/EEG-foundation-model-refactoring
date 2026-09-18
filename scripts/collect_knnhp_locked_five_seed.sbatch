#!/usr/bin/env bash
#SBATCH --account=system
#SBATCH --partition=cpu_medium
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=outputs/knn37_local_search_20260916_1157/knnhp5-collect-%x-%A_%a.out
#SBATCH --error=outputs/knn37_local_search_20260916_1157/knnhp5-collect-%x-%A_%a.err
set -euo pipefail

cd /gpfs/data/oermannlab/users/ml10266/workspace/EEG-founation-model
slug="${1:?dataset slug}"
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec .venv/bin/python -m scripts.run_knnhp_locked_five_seed evaluate \
  --slug "$slug" --index "$SLURM_ARRAY_TASK_ID"
