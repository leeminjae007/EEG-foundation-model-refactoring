#!/usr/bin/env bash
#SBATCH --job-name=mjbrain-d400-verify
#SBATCH --account=system
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/dim400_downstream_20260917/pretrain-verify-%j.out
#SBATCH --error=outputs/dim400_downstream_20260917/pretrain-verify-%j.err
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/activate.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec .venv/bin/python -m scripts.verify_dim400_pretrain
