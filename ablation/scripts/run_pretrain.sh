#!/usr/bin/env bash
# Submit one arm or a group from the SSH login node, with four GPUs in 1-4 nodes.
# Usage: bash ablation/scripts/run_pretrain.sh [--replace-pending] ARM_OR_GROUP [SEED] [TRAIN_ARGS...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
replace_pending=0
if [[ "${1:-}" == --replace-pending ]]; then replace_pending=1; shift; fi
group="${1:-encoder}"
seed="${2:-42}"
if (( $# >= 2 )); then shift 2; else shift "$#"; fi
case "$group" in
  encoder) arms=(encoder_labram encoder_cbramod encoder_csbrain encoder_mjde encoder_mjde_lite) ;;
  pe) arms=(pe_none pe_channel_id pe_acpe pe_reve4d pe_shpe) ;;
  all) arms=(encoder_labram encoder_cbramod encoder_csbrain encoder_mjde encoder_mjde_lite pe_none pe_channel_id pe_acpe pe_reve4d pe_shpe) ;;
  encoder_labram|encoder_cbramod|encoder_csbrain|encoder_mjde|encoder_mjde_lite|pe_none|pe_channel_id|pe_acpe|pe_reve4d|pe_shpe) arms=("$group") ;;
  *) echo "Usage: $0 [--replace-pending] {encoder|pe|all|CONFIG_NAME} [SEED] [TRAIN_ARGS...]" >&2; exit 2 ;;
esac
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "SEED must be a nonnegative integer" >&2; exit 2; }
mkdir -p logs/ablation
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1

for arm in "${arms[@]}"; do
  if [[ "$arm" == encoder_* ]]; then
    job_name="enc-${arm#encoder_}-s${seed}"
  else
    job_name="pe-${arm#pe_}-s${seed}"
  fi
  queued="$(squeue --noheader --user="$USER" --name="$job_name" --format='%i %T')"
  if (( replace_pending )) && [[ -n "$queued" ]]; then
    while read -r job_id state; do
      if [[ "$state" == PENDING ]]; then
        # Recheck state at cancellation time so a job that just started survives.
        scancel --state=PENDING "$job_id"
      fi
    done <<< "$queued"
    queued="$(squeue --noheader --user="$USER" --name="$job_name" --format='%i %T')"
  fi
  if [[ -n "$queued" ]]; then
    echo "Skip $job_name: already queued/running ($queued)"
    continue
  fi
  sbatch --job-name="$job_name" --chdir="$PWD" \
    ablation/scripts/pretrain_flexible.slurm "$arm" "$seed" "$@"
done
