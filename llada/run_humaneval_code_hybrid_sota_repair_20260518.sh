#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <gpu> <block_length>"
  echo "Supported block_length: 16 or 64"
  exit 1
fi

GPU="$1"
BLOCK_LENGTH="$2"

case "${BLOCK_LENGTH}" in
  16)
    EXTRA_ARGS="boundary_prior_threshold=0.7,semantic_min_block_length=8,semantic_selection_mode=max_score_above_threshold,boundary_prior_weight=0.3,delimiter_threshold=0.3"
    ;;
  64)
    EXTRA_ARGS="boundary_prior_threshold=0.6,semantic_min_block_length=16,semantic_selection_mode=max_score_above_threshold,boundary_prior_weight=0.5,delimiter_threshold=0.3"
    ;;
  *)
    echo "Unsupported block_length: ${BLOCK_LENGTH}"
    exit 1
    ;;
esac

exec bash /home/nvme01/workspace/AdaBlock-dLLM-main/llada/run_code_eval_config.sh \
  "${GPU}" \
  humaneval \
  semantic_hybrid \
  "${BLOCK_LENGTH}" \
  on \
  164 \
  b016_b064_sota_repair_20260518 \
  "${EXTRA_ARGS}"
