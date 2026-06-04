# Data Construction and Boundary Labels

This document describes the data path used by the three SemBlock boundary-head
lines.

## Common Label Format

The training scripts consume JSONL examples that contain text/code/math
segments and token-level boundary supervision. Unless a task-specific builder
states otherwise, each semantic segment contributes:

```text
last token of the segment -> boundary label 1
all other tokens          -> boundary label 0
```

This matches the runtime interpretation: a boundary head estimates whether a
candidate token position is a good place to end the current denoising block.

## Natural Language / GUM Line

Purpose: learn natural-language semantic/discourse boundaries for IFEval-style
instruction following.

Relevant code:

```text
llada/prepare_boundary_corpora.py
llada/build_boundary_gate_dataset.py
llada/train_boundary_segmenter.py
```

Dataset source: GUM discourse data. The processed split used by the local runs
was under:

```text
llada/data/semantic_boundary/processed/gum/{train,valid,test}.jsonl
```

Full processed data files are intentionally not committed. Rebuild them with
the scripts above or place equivalent JSONL files at the same paths.

## Math Line

Purpose: learn math-solution step boundaries used by GSM8K and Hendrycks MATH.

Relevant code:

```text
llada/build_task_conditioned_phase_boundary_jsonl.py
llada/math_oracle_utils.py
llada/math_oracle_benchmark.py
llada/train_boundary_segmenter.py
llada/rescore_hendrycks_math_cache.py
```

Typical processed locations used during development:

```text
llada/data/semantic_boundary/processed/math_v2_full_*/
llada/data/semantic_boundary/processed/combined/
```

For GSM8K/MATH inference, the SOTA scheduler combines the learned math head
with newline delimiter cues:

```text
block_strategy=semantic_hybrid
delimiter_ids=198
delimiter_threshold=0.3
boundary_prior_threshold=0.60
boundary_prior_weight=0.70
semantic_min_block_length=8
semantic_selection_mode=max_score_above_threshold
```

## Code / HumanEval Line

Purpose: learn code-phase boundaries without training on HumanEval benchmark
solutions.

Relevant code:

```text
llada/build_leetcode_humaneval_bridge_jsonl.py
llada/build_targeted_nonhe_aug.py
llada/codecontests_functionization.py
llada/label_phase_boundary_python.py
llada/train_boundary_segmenter.py
llada/analyze_humaneval_failures.py
```

The code SOTA head was trained on non-HumanEval code data and targeted
augmentation. HumanEval was used only for evaluation.

Typical processed locations used during development:

```text
llada/data/semantic_boundary/processed/nonhumaneval_code_leetcode_aug_failuretargeted_20260511/
llada/data/semantic_boundary/processed/leetcode_humaneval_bridge_*/
```

Full generated JSONL files are excluded from GitHub. The builders above
document the reconstruction logic.

