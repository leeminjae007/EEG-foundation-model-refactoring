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
  encoder) arms=(encoder_labram encoder_cbramod encoder_csbrain encoder_mjde encoder_mjde_lite encoder_mjde_s2t6 encoder_mjde_t2s6 encoder_mjde_average) ;;
  pe) arms=(pe_none pe_channel_id pe_acpe pe_reve4d pe_shpe) ;;
  all) arms=(encoder_labram encoder_cbramod encoder_csbrain encoder_mjde encoder_mjde_lite encoder_mjde_s2t6 encoder_mjde_t2s6 encoder_mjde_average pe_none pe_channel_id pe_acpe pe_reve4d pe_shpe) ;;
  encoder_labram|encoder_cbramod|encoder_csbrain|encoder_mjde|encoder_mjde_lite|encoder_mjde_s2t6|encoder_mjde_t2s6|encoder_mjde_average|encoder_mjde_mix1only|pe_none|pe_channel_id|pe_acpe|pe_reve4d|pe_shpe) arms=("$group") ;;
  *) echo "Usage: $0 [--replace-pending] {encoder|pe|all|CONFIG_NAME} [SEED] [TRAIN_ARGS...]" >&2; exit 2 ;;
esac
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "SEED must be a nonnegative integer" >&2; exit 2; }
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1

# Use the centrally maintained A100 health exclusion list.  The ablation
# launcher used to omit this when calling sbatch, allowing known-bad GPUs to
# be allocated despite their presence in the cluster policy.
excluded_nodes_csv="$(python - <<'PY'
from pathlib import Path
import yaml
policy = yaml.safe_load(Path("configs/cluster/bigpurple_a100.yaml").read_text())
print(",".join(policy["slurm"]["pretrain"].get("excluded_nodes", [])))
PY
)"
exclude_args=()
if [[ -n "$excluded_nodes_csv" ]]; then
  exclude_args=("--exclude=$excluded_nodes_csv")
fi

# A collaborator may execute this shared checkout without write access to the
# repository.  When --output is supplied, keep Slurm's stdout/stderr beside
# that owned experiment output rather than under the shared checkout's logs/.
train_args=("$@")
output_dir=""
for ((arg_index = 0; arg_index < ${#train_args[@]}; arg_index++)); do
  case "${train_args[$arg_index]}" in
    --output)
      ((arg_index + 1 < ${#train_args[@]})) || { echo "--output requires a directory" >&2; exit 2; }
      output_dir="${train_args[$((arg_index + 1))]}" ;;
    --output=*) output_dir="${train_args[$arg_index]#--output=}" ;;
  esac
done

for arm in "${arms[@]}"; do
  if [[ "$arm" == encoder_* ]]; then
    job_name="enc-${arm#encoder_}-s${seed}"
    if [[ "$arm" == encoder_mjde_mix1only ]]; then job_name="enc-mix1only-s${seed}"; fi
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
  log_args=()
  if [[ -n "$output_dir" ]]; then
    mkdir -p "$output_dir/logs"
    log_args=("--output=$output_dir/logs/%x-%j.out" "--error=$output_dir/logs/%x-%j.err")
  else
    mkdir -p logs/ablation
  fi
  sbatch --job-name="$job_name" --chdir="$PWD" "${exclude_args[@]}" "${log_args[@]}" \
    ablation/scripts/pretrain_flexible.slurm "$arm" "$seed" "$@"
done
