#!/usr/bin/env bash
#SBATCH --job-name=mjbrain-d400
#SBATCH --account=system
#SBATCH --partition=a100_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/pretrain/%x-%j.out
#SBATCH --error=logs/pretrain/%x-%j.err
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/activate.sh
export TMPDIR="${SLURM_TMPDIR:-/tmp}/mjbrain-d400-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

exec python -m torch.distributed.run --standalone --nproc_per_node=4 \
  pretrain.py --config configs/pretrain_dim400.yaml --distributed
