#!/usr/bin/env bash
# Called by srun once per GPU; the existing engine consumes torchrun-style env.
set -euo pipefail
: "${SLURM_PROCID:?Launch this worker through pretrain_flexible.slurm}"
: "${SLURM_NTASKS:?Missing Slurm task count}"
: "${SLURM_JOB_ID:?Missing Slurm job ID}"
: "${MASTER_ADDR:?Missing shared rendezvous host}"
: "${MASTER_PORT:?Missing shared rendezvous port}"

case "${CUDA_VISIBLE_DEVICES:-}" in
  ''|*,*|-1)
    echo "Expected one visible GPU per task; use srun --gpus-per-task=a100:1 --gpu-bind=single:1" >&2
    exit 2 ;;
esac
export RANK="$SLURM_PROCID"
export WORLD_SIZE="$SLURM_NTASKS"
export ABLATION_AUTO_RESUME="${ABLATION_AUTO_RESUME:-1}"
# Slurm exposes exactly one GPU to each task, which PyTorch sees as cuda:0.
# SLURM_LOCALID is NOT a CUDA index under this per-task binding.
export LOCAL_RANK=0
export TMPDIR="${SLURM_TMPDIR:-/tmp}/eeg-ablation-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
echo "rank=$RANK/$WORLD_SIZE node=${SLURMD_NODENAME:-unknown} gpu=$CUDA_VISIBLE_DEVICES local_rank=$LOCAL_RANK"
exec python -u -m ablation.pretrain "$@"
