#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_REPO="$(cd "${ROOT_DIR}/../../../.." && pwd)"
PYTHON_BIN="/home/nvme03/envs/DLLM/bin/python"
MODEL_PATH="/home/nvme03/workspace/models/GSAI-ML/LLaDA-8B-Instruct"
MATH_HEAD_PATH="/home/nvme01/workspace/AdaBlock-dLLM-main/llada/checkpoints/math_external_aqua_only_head_20260428/boundary_head_last.pt"

LIMIT="${LIMIT:-300}"
ALLOWED_NPUS="${ALLOWED_NPUS:-0,1}"
SLEEP_SECONDS="${SLEEP_SECONDS:-180}"

export HF_HOME="${ROOT_DIR}/cache/hf"
export HF_DATASETS_CACHE="${ROOT_DIR}/cache/hf/datasets"
export TRANSFORMERS_CACHE="${ROOT_DIR}/cache/hf/transformers"
export HF_ALLOW_CODE_EVAL=1
export NLTK_DATA="/home/nvme04/unknow/nltk_data"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/results" "${ROOT_DIR}/resume" "${ROOT_DIR}/traces" "${HF_HOME}"

wait_for_npu() {
  local npu="${ALLOWED_NPUS%%,*}"
  if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info -i "${npu}" >&2 || npu-smi info >&2 || true
  fi
  echo "${npu}"
}

latest_result() {
  local result_dir="$1"
  find "${result_dir}" -name 'results_*.json' -type f 2>/dev/null | sort | tail -1
}

run_variant() {
  local name="$1"
  local model_args="$2"
  local result_dir="${ROOT_DIR}/results/${name}"
  local trace_dir="${ROOT_DIR}/traces/${name}"
  local save_dir="${ROOT_DIR}/resume/${name}"

  if [[ -n "$(latest_result "${result_dir}")" ]]; then
    echo "[skip] ${name}: result already exists."
    return 0
  fi

  mkdir -p "${result_dir}" "${trace_dir}" "${save_dir}"
  local npu
  npu="$(wait_for_npu)"
  echo "[run] ${name}: npu=${npu}, limit=${LIMIT}"
  (
    cd "${CURRENT_REPO}/llada"
    ASCEND_RT_VISIBLE_DEVICES="${npu}" NPU_VISIBLE_DEVICES="${npu}" "${PYTHON_BIN}" "${CURRENT_REPO}/llada/eval_llada_adablock.py" \
      --model llada_dist \
      --model_args "${model_args},trace_dir=${trace_dir},save_dir=${save_dir}" \
      --tasks gsm8k \
      --num_fewshot 5 \
      --confirm_run_unsafe_code \
      --limit "${LIMIT}" \
      --batch_size 1 \
      --output_path "${result_dir}"
  ) > "${ROOT_DIR}/logs/${name}.log" 2>&1
  "${PYTHON_BIN}" "${ROOT_DIR}/summarize_gsm8k_sota_boundary_ablation.py"
}

BASE_ARGS="model_path=${MODEL_PATH},block_length=32,steps=16,gen_length=512,threshold=0.9,task_type=math,show_speed=True,use_cache=True,delimiter_ids=198,delimiter_threshold=0.3,gsm8k_landing_control=true"

# SOTA reference is not re-run here; summarize_gsm8k_sota_boundary_ablation.py
# reads the already-completed limit-300 SOTA result.
run_variant "delimiter_only_adablock" \
  "${BASE_ARGS},block_strategy=adablock"

run_variant "math_head_only" \
  "${BASE_ARGS},block_strategy=semantic_hybrid,boundary_prior_path=${MATH_HEAD_PATH},boundary_prior_threshold=0.60,boundary_prior_weight=1.0,semantic_min_block_length=8,semantic_selection_mode=max_score_above_threshold"

"${PYTHON_BIN}" "${ROOT_DIR}/summarize_gsm8k_sota_boundary_ablation.py"
echo "GSM8K SOTA boundary ablation finished."
