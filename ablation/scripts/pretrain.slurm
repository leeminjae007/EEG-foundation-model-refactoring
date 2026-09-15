#!/usr/bin/env bash
#SBATCH --job-name=eeg-ablation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ablation/%x-%j.out
#SBATCH --error=logs/ablation/%x-%j.err
set -euo pipefail

# Submit from the project root; choose partition/account on the sbatch command.
cd "$SLURM_SUBMIT_DIR"
source scripts/activate.sh
export TMPDIR="${SLURM_TMPDIR:-/tmp}/eeg-ablation-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
arm="${1:-encoder_labram}"
seed="${2:-42}"
python -m torch.distributed.run --standalone --nproc_per_node=4 --module ablation.pretrain \
  --config "ablation/configs/${arm}.yaml" --distributed --seed "$seed"
