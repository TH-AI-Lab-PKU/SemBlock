# Code Manifest

This manifest summarizes the files that matter for reproducing the SOTA and
ablation experiments.

## Data and Label Construction

```text
llada/prepare_boundary_corpora.py
llada/build_boundary_gate_dataset.py
llada/build_task_conditioned_phase_boundary_jsonl.py
llada/build_leetcode_humaneval_bridge_jsonl.py
llada/build_targeted_nonhe_aug.py
llada/build_local_boundary_correction_dataset.py
llada/label_phase_boundary_python.py
llada/mix_semantic_boundary_jsonl.py
```

## Training

```text
llada/train_boundary_segmenter.py
llada/train_local_boundary_corrector.py
llada/semantic_task_conditioned_head.py
llada/model/
llada/models/local_boundary_corrector.py
```

## Runtime and Evaluation

```text
llada/eval_llada_adablock.py
llada/eval_llada_semantic.py
llada/generate_semantic.py
llada/semantic_boundary.py
llada/semantic_scheduler.py
llada/semantic_runtime_carry.py
llada/semantic_runtime_hybrid.py
llada/semantic_runtime_length.py
llada/semantic_runtime_stateful.py
llada/gsm8k_landing.py
```

## Task-specific Evaluation

```text
llada/run_ifeval_gsm8k_queue_20260516.sh
llada/run_hendrycks_math_b016_b064_20260516.sh
llada/run_humaneval_code_sota_exact_20260518.sh
llada/run_opencompass_llada_1p5_humaneval_b016_b064_20260519.sh
llada/opencompass_llada_1p5_humaneval_b16_confidence.py
llada/opencompass_llada_1p5_humaneval_b64_confidence.py
```

## Ablations and Post-processing

```text
llada/compare_boundary_traces.py
llada/rescore_hendrycks_math_cache.py
llada/experiments/semantic_boundary_indep/20260516_boundary_overlap_full_corresponding/
llada/experiments/semantic_boundary_indep/20260518_ifeval_boundary_naturalness_curve/
llada/experiments/semantic_boundary_indep/20260518_ifeval_semantic_degradation_curve/
llada/experiments/semantic_boundary_indep/20260519_gsm8k_sota_boundary_ablation/
```
