#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_REPO="$(cd "${ROOT_DIR}/../../../.." && pwd)"
OLD_LLADA="/home/nvme01/workspace/AdaBlock-dLLM-main/llada"
PYTHON_BIN="/home/nvme03/envs/DLLM/bin/python"
MODEL_PATH="/home/nvme03/workspace/models/GSAI-ML/LLaDA-8B-Instruct"
TASK_PATH="${OLD_LLADA}/eval_tasks"
GUM_HEAD_PATH="${OLD_LLADA}/checkpoints/gum_direct_20260413/boundary_head_best.pt"

SOURCE_ROOT="${CURRENT_REPO}/llada/experiments/semantic_boundary_indep/20260516_boundary_overlap_tenth_corresponding"
SOURCE_ADABLOCK_TRACE="${SOURCE_ROOT}/traces/ifeval/adablock_tenth"
SOURCE_SEMANTIC_TRACE="${SOURCE_ROOT}/traces/ifeval/gum_head_tenth"

LIMIT="${LIMIT:-55}"
ALLOWED_GPUS="${ALLOWED_GPUS:-2,3,4,5,6,7}"
MIN_FREE_MB="${MIN_FREE_MB:-65000}"
MAX_UTIL="${MAX_UTIL:-20}"
SLEEP_SECONDS="${SLEEP_SECONDS:-180}"

export HF_HOME="${ROOT_DIR}/cache/hf"
export HF_DATASETS_CACHE="${ROOT_DIR}/cache/hf/datasets"
export TRANSFORMERS_CACHE="${ROOT_DIR}/cache/hf/transformers"
export HF_ALLOW_CODE_EVAL=1
export NLTK_DATA="/home/nvme04/unknow/nltk_data"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/results" "${ROOT_DIR}/resume" "${ROOT_DIR}/traces" "${ROOT_DIR}/overlap" "${HF_HOME}"

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 2
  fi
}

wait_for_gpu() {
  while true; do
    local gpu
    gpu="$(
      nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
        | awk -F, -v allowed="${ALLOWED_GPUS}" -v min_free="${MIN_FREE_MB}" -v max_util="${MAX_UTIL}" '
            BEGIN {
              split(allowed, ids, ",");
              for (i in ids) ok[ids[i] + 0] = 1;
            }
            {
              gsub(/ /, "", $1);
              gsub(/ /, "", $2);
              gsub(/ /, "", $3);
              idx = $1 + 0;
              free = $2 + 0;
              util = $3 + 0;
              if (ok[idx] && free >= min_free && util <= max_util) {
                print idx;
                exit;
              }
            }'
    )"
    if [[ -n "${gpu}" ]]; then
      echo "${gpu}"
      return 0
    fi
    date >&2
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits >&2
    echo "No allowed GPU (${ALLOWED_GPUS}) with >= ${MIN_FREE_MB} MB free and <= ${MAX_UTIL}% util; sleeping ${SLEEP_SECONDS}s." >&2
    sleep "${SLEEP_SECONDS}"
  done
}

latest_result() {
  local result_dir="$1"
  find "${result_dir}" -name 'results_*.json' -type f 2>/dev/null | sort | tail -1
}

run_eval() {
  local name="$1"
  local strategy="$2"
  local model_args="$3"
  local trace_dir="${ROOT_DIR}/traces/${name}"
  local save_dir="${ROOT_DIR}/resume/${name}"
  local result_dir="${ROOT_DIR}/results/${name}"
  local trace_file="${trace_dir}/rank_0.jsonl"

  if [[ -s "${trace_file}" ]] && [[ -n "$(latest_result "${result_dir}")" ]]; then
    echo "[skip] ${name}: trace and result already exist."
    return 0
  fi

  mkdir -p "${trace_dir}" "${save_dir}" "${result_dir}"
  local gpu
  gpu="$(wait_for_gpu)"
  echo "[run] ${name} (${strategy}) on GPU ${gpu}, limit=${LIMIT}"
  (
    cd "${OLD_LLADA}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${OLD_LLADA}/eval_llada_adablock.py" \
      --model llada_dist \
      --model_args "${model_args},trace_dir=${trace_dir},save_dir=${save_dir}" \
      --tasks ifeval_local \
      --include_path "${TASK_PATH}" \
      --limit "${LIMIT}" \
      --batch_size 1 \
      --output_path "${result_dir}"
  ) > "${ROOT_DIR}/logs/${name}.log" 2>&1
}

run_compare() {
  local name="$1"
  local trace_dir="$2"
  local prior_trace="${ROOT_DIR}/traces/prior_punctuation"
  if [[ ! -s "${trace_dir}/rank_0.jsonl" ]] || [[ ! -s "${prior_trace}/rank_0.jsonl" ]]; then
    echo "[skip] compare ${name}: missing trace."
    return 0
  fi
  "${PYTHON_BIN}" "${CURRENT_REPO}/llada/compare_boundary_traces.py" \
    --pair "ifeval_${name}_vs_prior" "${trace_dir}" "${prior_trace}" \
    --match-key sample_id \
    --output "${ROOT_DIR}/overlap/${name}_vs_prior_punctuation.json" \
    --per-sample-output "${ROOT_DIR}/overlap/${name}_vs_prior_punctuation.samples.jsonl"
}

require_path "${MODEL_PATH}"
require_path "${TASK_PATH}"
require_path "${GUM_HEAD_PATH}"
require_path "${SOURCE_ADABLOCK_TRACE}/rank_0.jsonl"
require_path "${SOURCE_SEMANTIC_TRACE}/rank_0.jsonl"

COMMON_ARGS="model_path=${MODEL_PATH},block_length=32,steps=16,gen_length=512,threshold=0.9,show_speed=True,use_cache=True"

run_eval "fixed" "fixed block" \
  "${COMMON_ARGS},block_strategy=fixed,delimiter_ids=198,11=1,13=1,delimiter_threshold=inf"

run_eval "prior_punctuation" "punctuation/sentence prior" \
  "${COMMON_ARGS},block_strategy=prior,task_type=nl,boundary_window_ratio=0.25"

run_compare "fixed" "${ROOT_DIR}/traces/fixed"
run_compare "adablock" "${SOURCE_ADABLOCK_TRACE}"
run_compare "semantic_head" "${SOURCE_SEMANTIC_TRACE}"

"${PYTHON_BIN}" "${ROOT_DIR}/summarize_ifeval_boundary_curve.py"

echo "IFEval boundary naturalness curve finished."
