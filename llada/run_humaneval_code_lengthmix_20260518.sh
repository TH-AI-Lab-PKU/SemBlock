#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <npu> <block_length> <shard_id>"
  exit 1
fi

NPU="$1"
BLOCK_LENGTH="$2"
SHARD_ID="$3"

REPO_DIR="/home/ubuntu/.config/superpowers/worktrees/AdaBlock-dLLM-main/semantic-boundary-indep-20260409/llada"
ACCELERATE_BIN="/home/nvme03/envs/DLLM/bin/accelerate"
MODEL_PATH="/home/nvme03/workspace/models/GSAI-ML/LLaDA-8B-Instruct"
TASK_PATH="${REPO_DIR}/lm_eval_tasks"
TASK_NAME="llada_humaneval_subset"
MANIFEST_ROOT="${REPO_DIR}/experiments/semantic_boundary_indep/20260511_full_humaneval_router_plus38_plus157_sharded/subset_manifests"
BOUNDARY_HEAD="${REPO_DIR}/checkpoints/nonhe_aug_failuretargeted_20260511/boundary_head_step_000250.pt"
BOUNDARY_THRESHOLD="${BOUNDARY_THRESHOLD:-0.5}"
BOUNDARY_WINDOW_RATIO="${BOUNDARY_WINDOW_RATIO:-0.25}"
CANDIDATE_BLOCK_LENGTHS="${CANDIDATE_BLOCK_LENGTHS:-16|32|64}"
MAX_BLOCK_LENGTH="${MAX_BLOCK_LENGTH:-64}"
RUN_TAG="${RUN_TAG:-lengthmix_b016_b064}"
RUN_ROOT="${REPO_DIR}/experiments/semantic_boundary_indep/20260518_humaneval_code_${RUN_TAG}"

run_dir="${RUN_ROOT}/b${BLOCK_LENGTH}/shard${SHARD_ID}"
log_path="${RUN_ROOT}/b${BLOCK_LENGTH}/shard${SHARD_ID}.log"
manifest="${MANIFEST_ROOT}/humaneval_full_shard${SHARD_ID}.json"

mkdir -p "${run_dir}/generations" "${run_dir}/traces"

cd "${REPO_DIR}"
PYTHONPATH="${REPO_DIR}" \
PYTHONNOUSERSITE=1 \
HF_HOME=/home/nvme03/hf \
HF_ALLOW_CODE_EVAL=1 \
HF_DATASETS_TRUST_REMOTE_CODE=true \
TOKENIZERS_PARALLELISM=false \
LLADA_HUMANEVAL_SUBSET_MANIFEST="${manifest}" \
ASCEND_RT_VISIBLE_DEVICES="${NPU}" \
NPU_VISIBLE_DEVICES="${NPU}" \
"${ACCELERATE_BIN}" launch --num_processes=1 eval_llada_semantic.py \
  --tasks "${TASK_NAME}" \
  --include_path "${TASK_PATH}" \
  --num_fewshot 0 \
  --confirm_run_unsafe_code \
  --model llada_semantic \
  --model_args "model_path=${MODEL_PATH},gen_length=512,steps=32,block_length=${BLOCK_LENGTH},threshold=0.9,boundary_head_path=${BOUNDARY_HEAD},boundary_threshold=${BOUNDARY_THRESHOLD},boundary_window_ratio=${BOUNDARY_WINDOW_RATIO},scheduler_variant=length_only,candidate_block_lengths=${CANDIDATE_BLOCK_LENGTHS},max_block_length=${MAX_BLOCK_LENGTH},phase_entropy_gate=0.8,transition_weight=0.0,runtime_mode=boundary_only,use_cache=False,show_speed=True,save_dir=${run_dir}/generations,trace_dir=${run_dir}/traces,phase_aware_transfer=False,boundary_guard=False,syntax_aware_landing=True,commit_reopen_tokens=8" \
  --output_path "${run_dir}" \
  --log_samples \
  > "${log_path}" 2>&1
