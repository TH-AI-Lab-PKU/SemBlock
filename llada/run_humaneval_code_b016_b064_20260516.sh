#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <npu> <shard_id>"
  exit 1
fi

NPU="$1"
SHARD_ID="$2"

REPO_DIR="/home/ubuntu/.config/superpowers/worktrees/AdaBlock-dLLM-main/semantic-boundary-indep-20260409/llada"
ACCELERATE_BIN="/home/nvme03/envs/DLLM/bin/accelerate"
MODEL_PATH="/home/nvme03/workspace/models/GSAI-ML/LLaDA-8B-Instruct"
TASK_PATH="${REPO_DIR}/lm_eval_tasks"
TASK_NAME="llada_humaneval_subset"
MANIFEST_ROOT="${REPO_DIR}/experiments/semantic_boundary_indep/20260511_full_humaneval_router_plus38_plus157_sharded/subset_manifests"
BOUNDARY_HEAD="${REPO_DIR}/checkpoints/nonhe_aug_failuretargeted_20260511/boundary_head_step_000250.pt"
RUN_ROOT="${REPO_DIR}/experiments/semantic_boundary_indep/20260516_humaneval_code_b016_b064"

run_block() {
  local block_length="$1"
  local run_dir="${RUN_ROOT}/b${block_length}/shard${SHARD_ID}"
  local log_path="${RUN_ROOT}/b${block_length}/shard${SHARD_ID}.log"
  local manifest="${MANIFEST_ROOT}/humaneval_full_shard${SHARD_ID}.json"

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
    --model_args "model_path=${MODEL_PATH},gen_length=512,steps=32,block_length=${block_length},threshold=0.9,boundary_head_path=${BOUNDARY_HEAD},boundary_threshold=0.5,boundary_window_ratio=0.125,scheduler_variant=failure_case_router,phase_entropy_gate=0.8,transition_weight=0.0,runtime_mode=boundary_only,use_cache=False,show_speed=True,save_dir=${run_dir}/generations,trace_dir=${run_dir}/traces,phase_aware_transfer=False,boundary_guard=False,syntax_aware_landing=True,commit_reopen_tokens=8" \
    --output_path "${run_dir}" \
    --log_samples \
    > "${log_path}" 2>&1
}

run_block 16
run_block 64
