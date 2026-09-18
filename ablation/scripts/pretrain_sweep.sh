#!/usr/bin/env bash
set -euo pipefail

# Run inside an allocated GPU node: bash ablation/scripts/pretrain_sweep.sh encoder 4 /path/to/tueg 42
group="${1:-encoder}"
gpus="${2:-1}"
data_dir="${3:-}"
seed="${4:-42}"
case "$group" in
  encoder) arms=(encoder_labram encoder_cbramod encoder_csbrain encoder_mjde encoder_mjde_lite encoder_mjde_s2t6 encoder_mjde_t2s6 encoder_mjde_average) ;;
  pe) arms=(pe_none pe_channel_id pe_acpe pe_reve4d pe_shpe) ;;
  all) arms=(encoder_labram encoder_cbramod encoder_csbrain encoder_mjde encoder_mjde_lite encoder_mjde_s2t6 encoder_mjde_t2s6 encoder_mjde_average pe_none pe_channel_id pe_acpe pe_reve4d pe_shpe) ;;
  *) echo "Usage: $0 {encoder|pe|all} GPU_COUNT [TUEG_LMDB] [SEED]" >&2; exit 2 ;;
esac
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
extra=()
if [[ -n "$data_dir" ]]; then extra+=(--data-dir "$data_dir"); fi
mkdir -p logs/ablation
for arm in "${arms[@]}"; do
  python -m torch.distributed.run --standalone --nproc_per_node="$gpus" --module ablation.pretrain \
    --config "ablation/configs/$arm.yaml" --distributed --seed "$seed" "${extra[@]}" \
    2>&1 | tee "logs/ablation/${arm}_seed${seed}_$(date +%Y%m%d_%H%M%S).log"
done
