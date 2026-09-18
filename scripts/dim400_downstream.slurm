#!/usr/bin/env bash
#SBATCH --job-name=mjbrain-d400-ds10
#SBATCH --account=system
#SBATCH --partition=gl40s_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=outputs/dim400_downstream_20260917/slurm-%A_%a.out
#SBATCH --error=outputs/dim400_downstream_20260917/slurm-%A_%a.err
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/activate.sh
export TMPDIR="${SLURM_TMPDIR:-/tmp}/mjbrain-d400-ds-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

exec srun --ntasks=1 --gpus-per-task=l40s:1 --gpu-bind=single:1 \
  --kill-on-bad-exit=1 .venv/bin/python -m scripts.run_dim400_downstream_array
