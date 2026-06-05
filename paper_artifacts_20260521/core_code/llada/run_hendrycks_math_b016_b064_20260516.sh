#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <npu> <block_length> [wait_for_tmux_session]"
  exit 1
fi

NPU="$1"
BLOCK_LENGTH="$2"
WAIT_FOR_SESSION="${3:-}"

if [[ -n "${WAIT_FOR_SESSION}" ]]; then
  while tmux has-session -t "${WAIT_FOR_SESSION}" 2>/dev/null; do
    echo "$(date '+%F %T') waiting for tmux session ${WAIT_FOR_SESSION} before starting MATH B${BLOCK_LENGTH}"
    sleep 300
  done
fi

WAIT_FOR_NPU_READY="${WAIT_FOR_NPU_READY:-0}"
if [[ "${WAIT_FOR_NPU_READY}" == "1" && -x "$(command -v npu-smi 2>/dev/null)" ]]; then
  npu-smi info -i "${NPU}" >/dev/null 2>&1 || npu-smi info >/dev/null 2>&1 || true
fi

REPO_DIR="/home/nvme01/workspace/AdaBlock-dLLM-main/llada"
WORKTREE_DIR="/home/ubuntu/.config/superpowers/worktrees/AdaBlock-dLLM-main/semantic-boundary-indep-20260409"
PYTHON_BIN="/home/nvme03/envs/DLLM/bin/python"
MODEL_PATH="/home/nvme03/workspace/models/GSAI-ML/LLaDA-8B-Instruct"
TAG="aqua_hendrycks_math_b016_b064_20260516"
TASK_NAME="hendrycks_math"
STRATEGY="semantic_hybrid"
RUN_SUFFIX="_thr0p60_minb8_selmax_score_above_threshold_mix0p70_dthr0p30_landfalse"
RUN_NAME="${STRATEGY}_b${BLOCK_LENGTH}_cache_on_full${RUN_SUFFIX}"

RESULT_ROOT="${REPO_DIR}/eval_results_math_semantic/${TAG}/${TASK_NAME}"
LOG_ROOT="${REPO_DIR}/logs/math_semantic/${TAG}/${TASK_NAME}"
RESUME_ROOT="${REPO_DIR}/resume_cache_math/${TAG}/${TASK_NAME}"
OUT_PATH="${RESULT_ROOT}/${RUN_NAME}"
LOG_PATH="${LOG_ROOT}/${RUN_NAME}.log"
SAVE_DIR="${RESUME_ROOT}/${RUN_NAME}"
OFFLINE_JSON="${RESULT_ROOT}/${RUN_NAME}_offline_rescore.json"

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}" "${RESUME_ROOT}"

MODEL_ARGS="model_path=${MODEL_PATH},block_strategy=${STRATEGY},block_length=${BLOCK_LENGTH},steps=16,gen_length=512,threshold=0.9,task_type=math,show_speed=True,use_cache=True,delimiter_ids=198,boundary_prior_path=${REPO_DIR}/checkpoints/math_external_aqua_only_head_20260428/boundary_head_last.pt,boundary_prior_threshold=0.60,semantic_min_block_length=8,semantic_selection_mode=max_score_above_threshold,boundary_prior_weight=0.7,delimiter_threshold=0.3,gsm8k_landing_control=false,save_dir=${SAVE_DIR}"

cd "${REPO_DIR}"

echo "$(date '+%F %T') starting ${RUN_NAME} on NPU ${NPU}"
HF_ALLOW_CODE_EVAL=1 \
HF_DATASETS_TRUST_REMOTE_CODE=true \
PYTHONNOUSERSITE=1 \
ASCEND_RT_VISIBLE_DEVICES="${NPU}" \
NPU_VISIBLE_DEVICES="${NPU}" \
"${PYTHON_BIN}" "${REPO_DIR}/eval_llada_adablock.py" \
  --model llada_dist \
  --tasks "${TASK_NAME}" \
  --confirm_run_unsafe_code \
  --model_args "${MODEL_ARGS}" \
  --batch_size 1 \
  --output_path "${OUT_PATH}" \
  > "${LOG_PATH}" 2>&1

echo "$(date '+%F %T') scoring ${RUN_NAME}"
PYTHONNOUSERSITE=1 \
"${PYTHON_BIN}" "${WORKTREE_DIR}/llada/rescore_hendrycks_math_cache.py" \
  --cache-jsonl "${SAVE_DIR}/rank_0.jsonl" \
  --output-json "${OFFLINE_JSON}" \
  >> "${LOG_PATH}" 2>&1

echo "$(date '+%F %T') done ${RUN_NAME}; offline rescore: ${OFFLINE_JSON}"
