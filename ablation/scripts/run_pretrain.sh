#!/usr/bin/env bash
# Submit one arm or a group from the SSH login node, with four GPUs in 1-4 nodes.
# Usage: bash ablation/scripts/run_pretrain.sh [--replace-pending] [--pretrain-only] ARM_OR_GROUP [SEED] [TRAIN_ARGS...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
replace_pending=0
pretrain_only=0
if [[ "${1:-}" == --replace-pending ]]; then replace_pending=1; shift; fi
if [[ "${1:-}" == --pretrain-only ]]; then pretrain_only=1; shift; fi
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
  if [[ -n "$output_dir" && "$pretrain_only" == 0 ]]; then
    mkdir -p "$output_dir/logs"
    log_args=("--output=$output_dir/logs/%x-%j.out" "--error=$output_dir/logs/%x-%j.err")
  else
    mkdir -p logs/ablation
  fi
  pretrain_job="$(sbatch --parsable --job-name="$job_name" --chdir="$PWD" "${exclude_args[@]}" "${log_args[@]}" \
    ablation/scripts/pretrain_flexible.slurm "$arm" "$seed" "$@")"
  pretrain_job="${pretrain_job%%;*}"
  echo "Submitted pretrain $pretrain_job ($job_name)"
  if [[ -n "$output_dir" ]]; then
    experiment_dir="$(dirname "$output_dir")"
    checkpoint="$output_dir/checkpoint-epoch-0040.pth"
    python - "$experiment_dir" "$arm" "$pretrain_job" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
path = root / "manifest.json"
if not path.exists():
    path.write_text(json.dumps({"ablation_arm": sys.argv[2], "pretrain_job": sys.argv[3]}, indent=2) + "\n")
PY
    callback="$(sbatch --parsable --account=system --job-name="${job_name}-downstream-dispatch" \
      --partition=cpu_short,cpu_long --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=01:00:00 \
      --dependency="afterok:$pretrain_job" --kill-on-invalid-dep=yes --chdir="$PWD" \
      --output="$experiment_dir/downstream-dispatch-%j.out" --error="$experiment_dir/downstream-dispatch-%j.err" \
      --wrap="exec $(command -v python) scripts/submit_existing_downstream.py --experiment $experiment_dir --checkpoint $checkpoint --l40s")"
    echo "Submitted downstream dispatcher ${callback%%;*} (afterok:$pretrain_job)"
  fi
done
