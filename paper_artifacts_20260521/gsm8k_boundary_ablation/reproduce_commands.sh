#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root:
# /home/ubuntu/.config/superpowers/worktrees/AdaBlock-dLLM-main/semantic-boundary-indep-20260409

EXP_DIR="llada/experiments/semantic_boundary_indep/20260519_gsm8k_sota_boundary_ablation"

# The launcher reads the existing SOTA limit-300 reference and runs the two ablations.
# Keep GPU constraints explicit if re-running on a shared machine.
ALLOWED_GPUS=0,1 LIMIT=300 "${EXP_DIR}/run_gsm8k_sota_boundary_ablation.sh"

# Rebuild the summary table from result JSON files and logs.
/home/nvme03/envs/DLLM/bin/python "${EXP_DIR}/summarize_gsm8k_sota_boundary_ablation.py"

