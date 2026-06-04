#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <gpu> <block_list_comma> <shard_id> [shard_id...]"
  exit 1
fi

GPU="$1"
BLOCK_LIST="$2"
shift 2

WORKTREE_DIR="/home/ubuntu/.config/superpowers/worktrees/AdaBlock-dLLM-main/semantic-boundary-indep-20260409"
RUNNER="${WORKTREE_DIR}/llada/run_humaneval_code_sota_exact_20260518.sh"
RUN_ROOT="${WORKTREE_DIR}/llada/experiments/semantic_boundary_indep/20260518_humaneval_code_sota_exact_b016_b064"

IFS=',' read -r -a BLOCKS <<< "${BLOCK_LIST}"

for shard_id in "$@"; do
  for block_length in "${BLOCKS[@]}"; do
    out_file="${RUN_ROOT}/b${block_length}/shard${shard_id}/generations/rank_0.jsonl"
    if [[ -f "${out_file}" ]] && [[ "$(wc -l < "${out_file}")" -ge 41 ]]; then
      echo "$(date '+%F %T') skip complete b${block_length} shard${shard_id}: ${out_file}"
      continue
    fi
    echo "$(date '+%F %T') run exact SOTA b${block_length} shard${shard_id} on GPU ${GPU}"
    bash "${RUNNER}" "${GPU}" "${block_length}" "${shard_id}"
  done
done
