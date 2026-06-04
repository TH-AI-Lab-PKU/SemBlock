#!/usr/bin/env bash
set -euo pipefail

OPENCOMPASS_DIR="/home/nvme02/workspace/wm_mem/wm/opencompass"
WORKTREE_DIR="/home/ubuntu/.config/superpowers/worktrees/AdaBlock-dLLM-main/semantic-boundary-indep-20260409"
PYTHON_BIN="/home/nvme03/envs/DLLM/bin/python"
PYTHONPATH_EXTRA="/home/nvme02/workspace/wm_mem/wm/human-eval"
OUT_DIR="${WORKTREE_DIR}/llada/opencompass_outputs/llada_1p5_humaneval_b016_b064_20260519"
LOG_DIR="${WORKTREE_DIR}/llada/logs"
LOG_FILE="${LOG_DIR}/llada_1p5_humaneval_b016_b064_$(date '+%Y%m%d_%H%M%S').log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "$(date '+%F %T') logging to ${LOG_FILE}"

cd "${OPENCOMPASS_DIR}"
export PYTHONPATH="${OPENCOMPASS_DIR}:${PYTHONPATH_EXTRA}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

echo "$(date '+%F %T') starting LLaDA-1.5 HumanEval B0=16 on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
"${PYTHON_BIN}" run.py "${WORKTREE_DIR}/llada/opencompass_llada_1p5_humaneval_b16_confidence.py" -w "${OUT_DIR}"

echo "$(date '+%F %T') starting LLaDA-1.5 HumanEval B0=64 on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
"${PYTHON_BIN}" run.py "${WORKTREE_DIR}/llada/opencompass_llada_1p5_humaneval_b64_confidence.py" -w "${OUT_DIR}"

echo "$(date '+%F %T') done LLaDA-1.5 HumanEval B0=16/64"
