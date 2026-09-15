#!/usr/bin/env bash
# Submit the full MJDE + REVE 4D PE experiment from the SSH login node.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
options=()
if [[ "${1:-}" == --replace-pending ]]; then options+=("$1"); shift; fi
exec bash ablation/scripts/run_pretrain.sh "${options[@]}" pe_reve4d 42 "$@"
