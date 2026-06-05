#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <npu> <block_length>"
  exit 1
fi

NPU="$1"
BLOCK_LENGTH="$2"
WORKTREE_DIR="/home/ubuntu/.config/superpowers/worktrees/AdaBlock-dLLM-main/semantic-boundary-indep-20260409"
MAIN_LLADA="/home/nvme01/workspace/AdaBlock-dLLM-main/llada"
WAIT="${WORKTREE_DIR}/llada/wait_for_gpu_then_run.sh"
MATH_HEAD="${MAIN_LLADA}/checkpoints/math_external_aqua_only_head_20260428/boundary_head_last.pt"
GSM8K_ARGS="boundary_prior_path=${MATH_HEAD},boundary_prior_threshold=0.60,boundary_prior_weight=0.70,semantic_min_block_length=8,semantic_selection_mode=max_score_above_threshold,delimiter_threshold=0.3,gsm8k_landing_control=true"

bash "${WAIT}" "${NPU}" env PYTHONNOUSERSITE=1 bash "${MAIN_LLADA}/run_ifeval_config.sh" \
  "${NPU}" semantic_head "${BLOCK_LENGTH}" on 541 \
  full_ifeval_gum_b016_b064_20260516 \
  boundary_prior_threshold=0.5

bash "${WAIT}" "${NPU}" env PYTHONNOUSERSITE=1 bash "${MAIN_LLADA}/run_math_eval_config.sh" \
  "${NPU}" gsm8k semantic_hybrid "${BLOCK_LENGTH}" on 1319 \
  aqua_gsm8k_b016_b064_20260516 \
  "${GSM8K_ARGS}"
