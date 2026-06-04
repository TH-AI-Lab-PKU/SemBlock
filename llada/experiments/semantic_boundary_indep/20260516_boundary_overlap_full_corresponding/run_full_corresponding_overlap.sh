#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_REPO="$(cd "${ROOT_DIR}/../../../.." && pwd)"
OLD_LLADA="/home/nvme01/workspace/AdaBlock-dLLM-main/llada"
PYTHON_BIN="/home/nvme03/envs/DLLM/bin/python"
MODEL_PATH="/home/nvme03/workspace/models/GSAI-ML/LLaDA-8B-Instruct"
CODE_HEAD_TRACE_DIR="${CURRENT_REPO}/llada/experiments/semantic_boundary_indep/SOTA_ARCHIVE_20260512_generalized_v3_4695/results/full_humaneval_router_generalized_v3_sharded"
GUM_HEAD_PATH="${OLD_LLADA}/checkpoints/gum_direct_20260413/boundary_head_best.pt"
MATH_HEAD_PATH="${OLD_LLADA}/checkpoints/math_external_aqua_only_head_20260428/boundary_head_last.pt"
TASK_PATH="${OLD_LLADA}/eval_tasks"

MIN_FREE_MB="${MIN_FREE_MB:-50000}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"

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
      nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F, -v min_free="${MIN_FREE_MB}" '
            {
              gsub(/ /, "", $1);
              gsub(/ /, "", $2);
              if ($2 + 0 >= min_free) {
                print $1;
                exit;
              }
            }'
    )"
    if [[ -n "${gpu}" ]]; then
      echo "${gpu}"
      return 0
    fi
    date >&2
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits >&2
    echo "No GPU with >= ${MIN_FREE_MB} MB free; sleeping ${SLEEP_SECONDS}s." >&2
    sleep "${SLEEP_SECONDS}"
  done
}

run_eval() {
  local name="$1"
  local repo_dir="$2"
  local trace_dir="$3"
  shift 3

  local trace_file="${trace_dir}/rank_0.jsonl"
  if [[ -s "${trace_file}" ]]; then
    echo "[skip] ${name}: trace already exists at ${trace_file}"
    return 0
  fi

  mkdir -p "${trace_dir}"
  local gpu
  gpu="$(wait_for_gpu)"
  echo "[run] ${name} on GPU ${gpu}"
  (
    cd "${repo_dir}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "$@"
  ) > "${ROOT_DIR}/logs/${name}.log" 2>&1
}

run_compare() {
  local task="$1"
  local semantic_dir="$2"
  local adablock_dir="$3"
  local output_name="$4"
  local output_path="${ROOT_DIR}/overlap/${output_name}.json"

  if [[ -s "${output_path}" ]]; then
    echo "[skip] compare ${task}: ${output_path}"
    return 0
  fi

  "${PYTHON_BIN}" "${CURRENT_REPO}/llada/compare_boundary_traces.py" \
    --pair "${task}" "${semantic_dir}" "${adablock_dir}" \
    --match-key sample_id \
    --output "${output_path}" \
    --per-sample-output "${ROOT_DIR}/overlap/${output_name}.samples.jsonl"
}

require_path "${MODEL_PATH}"
require_path "${CODE_HEAD_TRACE_DIR}"
require_path "${GUM_HEAD_PATH}"
require_path "${MATH_HEAD_PATH}"
require_path "${OLD_LLADA}/eval_llada_adablock.py"

echo "Experiment root: ${ROOT_DIR}"
echo "Pairing: HumanEval=Code head, IFEval=GUM head, GSM8K/MATH=Math head."

HUMANEVAL_ADA_TRACE="${ROOT_DIR}/traces/humaneval/adablock_full"
IFEVAL_SEM_TRACE="${ROOT_DIR}/traces/ifeval/gum_head_full"
IFEVAL_ADA_TRACE="${ROOT_DIR}/traces/ifeval/adablock_full"
GSM8K_SEM_TRACE="${ROOT_DIR}/traces/gsm8k/math_head_full"
GSM8K_ADA_TRACE="${ROOT_DIR}/traces/gsm8k/adablock_full"
MATH_SEM_TRACE="${ROOT_DIR}/traces/math/math_head_full"
MATH_ADA_TRACE="${ROOT_DIR}/traces/math/adablock_full"

run_eval "humaneval_adablock_full_trace" "${CURRENT_REPO}" "${HUMANEVAL_ADA_TRACE}" \
  "${CURRENT_REPO}/llada/eval_llada_adablock.py" \
  --model llada_dist \
  --model_args "model_path=${MODEL_PATH},gen_length=512,steps=32,block_length=32,threshold=0.9,delimiter_ids=198,delimiter_threshold=0.3,use_cache=True,dual_cache=True,show_speed=True,trace_dir=${HUMANEVAL_ADA_TRACE},save_dir=${ROOT_DIR}/resume/humaneval/adablock_full" \
  --tasks humaneval \
  --num_fewshot 0 \
  --confirm_run_unsafe_code \
  --batch_size 1 \
  --output_path "${ROOT_DIR}/results/humaneval/adablock_full" \
  --log_samples

run_compare "humaneval" "${CODE_HEAD_TRACE_DIR}" "${HUMANEVAL_ADA_TRACE}" "humaneval_code_vs_adablock_all"
"${PYTHON_BIN}" "${ROOT_DIR}/summarize_corresponding_overlap.py"

run_eval "ifeval_gum_head_full_trace" "${OLD_LLADA}" "${IFEVAL_SEM_TRACE}" \
  "${OLD_LLADA}/eval_llada_adablock.py" \
  --model llada_dist \
  --model_args "model_path=${MODEL_PATH},block_strategy=semantic_head,block_length=32,steps=16,gen_length=512,threshold=0.9,show_speed=True,use_cache=True,boundary_prior_path=${GUM_HEAD_PATH},boundary_prior_threshold=0.75,trace_dir=${IFEVAL_SEM_TRACE},save_dir=${ROOT_DIR}/resume/ifeval/gum_head_full" \
  --tasks ifeval_local \
  --include_path "${TASK_PATH}" \
  --batch_size 1 \
  --output_path "${ROOT_DIR}/results/ifeval/gum_head_full"

run_eval "ifeval_adablock_full_trace" "${OLD_LLADA}" "${IFEVAL_ADA_TRACE}" \
  "${OLD_LLADA}/eval_llada_adablock.py" \
  --model llada_dist \
  --model_args "model_path=${MODEL_PATH},block_strategy=adablock,block_length=32,steps=16,gen_length=512,threshold=0.9,show_speed=True,use_cache=True,delimiter_ids=198,11=1,13=1,delimiter_threshold=0.3,trace_dir=${IFEVAL_ADA_TRACE},save_dir=${ROOT_DIR}/resume/ifeval/adablock_full" \
  --tasks ifeval_local \
  --include_path "${TASK_PATH}" \
  --batch_size 1 \
  --output_path "${ROOT_DIR}/results/ifeval/adablock_full"

run_compare "ifeval" "${IFEVAL_SEM_TRACE}" "${IFEVAL_ADA_TRACE}" "ifeval_gum_vs_adablock_all"
"${PYTHON_BIN}" "${ROOT_DIR}/summarize_corresponding_overlap.py"

run_eval "gsm8k_math_head_full_trace" "${OLD_LLADA}" "${GSM8K_SEM_TRACE}" \
  "${OLD_LLADA}/eval_llada_adablock.py" \
  --model llada_dist \
  --model_args "model_path=${MODEL_PATH},block_strategy=semantic_hybrid,block_length=32,steps=16,gen_length=512,threshold=0.9,task_type=math,show_speed=True,use_cache=True,boundary_prior_path=${MATH_HEAD_PATH},boundary_prior_threshold=0.60,boundary_prior_weight=0.70,semantic_min_block_length=8,semantic_selection_mode=max_score_above_threshold,delimiter_ids=198,delimiter_threshold=0.3,gsm8k_landing_control=true,trace_dir=${GSM8K_SEM_TRACE},save_dir=${ROOT_DIR}/resume/gsm8k/math_head_full" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size 1 \
  --output_path "${ROOT_DIR}/results/gsm8k/math_head_full"

run_eval "gsm8k_adablock_full_trace" "${OLD_LLADA}" "${GSM8K_ADA_TRACE}" \
  "${OLD_LLADA}/eval_llada_adablock.py" \
  --model llada_dist \
  --model_args "model_path=${MODEL_PATH},block_strategy=adablock,block_length=32,steps=16,gen_length=512,threshold=0.9,task_type=math,show_speed=True,use_cache=True,delimiter_ids=198,delimiter_threshold=0.3,gsm8k_landing_control=true,trace_dir=${GSM8K_ADA_TRACE},save_dir=${ROOT_DIR}/resume/gsm8k/adablock_full" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size 1 \
  --output_path "${ROOT_DIR}/results/gsm8k/adablock_full"

run_compare "gsm8k" "${GSM8K_SEM_TRACE}" "${GSM8K_ADA_TRACE}" "gsm8k_math_vs_adablock_all"
"${PYTHON_BIN}" "${ROOT_DIR}/summarize_corresponding_overlap.py"

run_eval "math_head_full_trace" "${OLD_LLADA}" "${MATH_SEM_TRACE}" \
  "${OLD_LLADA}/eval_llada_adablock.py" \
  --model llada_dist \
  --model_args "model_path=${MODEL_PATH},block_strategy=semantic_hybrid,block_length=32,steps=16,gen_length=512,threshold=0.9,task_type=math,show_speed=True,use_cache=True,boundary_prior_path=${MATH_HEAD_PATH},boundary_prior_threshold=0.60,boundary_prior_weight=0.70,semantic_min_block_length=8,semantic_selection_mode=max_score_above_threshold,delimiter_ids=198,delimiter_threshold=0.3,gsm8k_landing_control=false,trace_dir=${MATH_SEM_TRACE},save_dir=${ROOT_DIR}/resume/math/math_head_full" \
  --tasks hendrycks_math \
  --batch_size 1 \
  --output_path "${ROOT_DIR}/results/math/math_head_full"

run_eval "math_adablock_full_trace" "${OLD_LLADA}" "${MATH_ADA_TRACE}" \
  "${OLD_LLADA}/eval_llada_adablock.py" \
  --model llada_dist \
  --model_args "model_path=${MODEL_PATH},block_strategy=adablock,block_length=32,steps=16,gen_length=512,threshold=0.9,task_type=math,show_speed=True,use_cache=True,delimiter_ids=198,delimiter_threshold=0.3,gsm8k_landing_control=false,trace_dir=${MATH_ADA_TRACE},save_dir=${ROOT_DIR}/resume/math/adablock_full" \
  --tasks hendrycks_math \
  --batch_size 1 \
  --output_path "${ROOT_DIR}/results/math/adablock_full"

run_compare "math" "${MATH_SEM_TRACE}" "${MATH_ADA_TRACE}" "math_math_vs_adablock_all"
"${PYTHON_BIN}" "${ROOT_DIR}/summarize_corresponding_overlap.py"

echo "All full corresponding boundary-overlap jobs finished."
