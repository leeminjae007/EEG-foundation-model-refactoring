#!/usr/bin/env bash
#SBATCH --job-name=knnhp-screen-test
#SBATCH --account=system
#SBATCH --partition=gl40s_dev,gl40s_short,gl40s_long
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-2%3
#SBATCH --output=outputs/knn37_local_search_20260916_1157/screen_test_slurm-%A_%a.log
#SBATCH --error=outputs/knn37_local_search_20260916_1157/screen_test_slurm-%A_%a.err
set -euo pipefail
cd /gpfs/data/oermannlab/users/ml10266/workspace/EEG-founation-model
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR="/tmp/knnhp-test-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
exec .venv/bin/python -m scripts.evaluate_knnhp_screen_leaders run --index "$SLURM_ARRAY_TASK_ID"
