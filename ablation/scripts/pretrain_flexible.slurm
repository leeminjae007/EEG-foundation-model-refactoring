#!/usr/bin/env bash
#SBATCH --job-name=eeg-ablation
#SBATCH --account=system
#SBATCH --partition=a100_short,a100_long
#SBATCH --nodes=1-4
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/ablation/%x-%j.out
#SBATCH --error=logs/ablation/%x-%j.err
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
# Submitters run the shared checkout from their own conda environment.  Slurm
# inherits that environment, so do not replace it with the checkout owner's
# optional .venv.  Retain the old path only for legacy owner-side submissions.
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  source scripts/activate.sh
else
  export PYTHONNOUSERSITE=1
fi
arm="${1:-encoder_labram}"
seed="${2:-42}"
if (( $# >= 2 )); then shift 2; else shift "$#"; fi

# One Slurm task per GPU, including uneven layouts such as 2+1+1 GPUs.
# All ranks rendezvous on the first allocated host, never on localhost.
hosts="$(scontrol show hostnames "$SLURM_JOB_NODELIST")"
export MASTER_ADDR="${hosts%%$'\n'*}"
export MASTER_PORT="${ABLATION_MASTER_PORT:-$((15000 + SLURM_JOB_ID % 40000))}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
export SLURM_EXPORT_ENV=ALL

echo "arm=$arm seed=$seed nodes=$SLURM_JOB_NODELIST ranks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT"
exec srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_NTASKS" \
  --gpus-per-task=a100:1 --gpu-bind=single:1 --distribution=block \
  --kill-on-bad-exit=1 --export=ALL \
  bash ablation/scripts/slurm_worker.sh \
  --config "ablation/configs/${arm}.yaml" --distributed --seed "$seed" "$@"
