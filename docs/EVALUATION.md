# SOTA Evaluation

This document lists the paper-facing SOTA evaluation entry points.

## IFEval

Model:

```text
/home/nvme03/workspace/models/GSAI-ML/LLaDA-8B-Instruct
```

SOTA B0=32 configuration:

```text
block_strategy=semantic_head
block_length=32
steps=16
gen_length=512
threshold=0.9
use_cache=True
boundary_prior_path=checkpoints/gum_direct_20260413/boundary_head_best.pt
boundary_prior_threshold=0.75
```

Runner template:

```text
llada/run_ifeval_gsm8k_queue_20260516.sh
```

## GSM8K

SOTA B0=32 configuration:

```text
block_strategy=semantic_hybrid
block_length=32
steps=16
gen_length=512
threshold=0.9
task_type=math
use_cache=True
num_fewshot=5
delimiter_ids=198
delimiter_threshold=0.3
gsm8k_landing_control=true
boundary_prior_path=checkpoints/math_external_aqua_only_head_20260428/boundary_head_last.pt
boundary_prior_threshold=0.60
boundary_prior_weight=0.70
semantic_min_block_length=8
semantic_selection_mode=max_score_above_threshold
```

## MATH

Runner:

```text
llada/run_hendrycks_math_b016_b064_20260516.sh
```

MATH uses the same math semantic-hybrid settings as GSM8K, with
`gsm8k_landing_control=false`. The reported paper value is from offline boxed
answer rescoring:

```text
llada/rescore_hendrycks_math_cache.py
```

## HumanEval

Code-head SOTA, LLaDA-Instruct B0=32:

```text
llada/eval_llada_semantic.py
llada/generate_semantic.py
llada/run_humaneval_code_sota_exact_20260518.sh
```

Main model args:

```text
gen_length=512
steps=32
block_length=32
threshold=0.9
boundary_head_path=checkpoints/nonhe_aug_failuretargeted_20260511/boundary_head_step_000250.pt
boundary_threshold=0.5
boundary_window_ratio=0.125
scheduler_variant=failure_case_router
phase_entropy_gate=0.8
transition_weight=0.0
runtime_mode=boundary_only
use_cache=False
syntax_aware_landing=True
commit_reopen_tokens=8
num_fewshot=0
```

LLaDA-1.5 HumanEval OpenCompass configs:

```text
llada/opencompass_llada_1p5_humaneval_b16_confidence.py
llada/opencompass_llada_1p5_humaneval_b64_confidence.py
llada/run_opencompass_llada_1p5_humaneval_b016_b064_20260519.sh
```

The B0=32 OpenCompass result is preserved in
`paper_artifacts_20260521/results_raw/humaneval_llada15_b32_summary.csv`.

## Result Tables

See:

```text
paper_artifacts_20260521/SOTA_RESULTS.md
paper_artifacts_20260521/SOTA_HYPERPARAMETERS.md
```

