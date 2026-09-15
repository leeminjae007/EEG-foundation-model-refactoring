#!/usr/bin/env bash
# Submit the full MJDE + REVE 4D PE experiment from the SSH login node.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p logs/ablation

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

exec sbatch \
  --account=system \
  --partition=a100_short,a100_long \
  --job-name=pe-reve4d-s42 \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:a100:4 \
  --cpus-per-task=32 \
  --mem=128G \
  --time=1-00:00:00 \
  --chdir="$PWD" \
  ablation/scripts/pretrain.slurm pe_reve4d 42
