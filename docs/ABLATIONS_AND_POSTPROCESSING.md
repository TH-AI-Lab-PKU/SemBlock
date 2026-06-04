# Ablations and Post-processing

This repository includes the code and compact result artifacts for the ablation
experiments used in the paper.

## Boundary Overlap with AdaBlock

Purpose: verify that learned semantic boundaries are not merely reproducing
AdaBlock delimiter-style boundaries.

Main code:

```text
llada/compare_boundary_traces.py
llada/experiments/semantic_boundary_indep/20260516_boundary_overlap_full_corresponding/run_full_corresponding_overlap.sh
llada/experiments/semantic_boundary_indep/20260516_boundary_overlap_full_corresponding/summarize_corresponding_overlap.py
```

Paper artifacts:

```text
paper_artifacts_20260521/boundary_overlap_with_adablock/
```

## IFEval Naturalness / Boundary Quality Curve

Purpose: evaluate whether more natural semantic boundaries correlate with
better instruction-following performance.

Code:

```text
llada/experiments/semantic_boundary_indep/20260518_ifeval_boundary_naturalness_curve/run_ifeval_boundary_curve.sh
llada/experiments/semantic_boundary_indep/20260518_ifeval_boundary_naturalness_curve/summarize_ifeval_boundary_curve.py
```

## IFEval Semantic Degradation

Purpose: perturb clean semantic boundaries with jitter/randomization and observe
performance degradation.

Code:

```text
llada/experiments/semantic_boundary_indep/20260518_ifeval_semantic_degradation_curve/run_ifeval_semantic_degradation_curve.sh
llada/experiments/semantic_boundary_indep/20260518_ifeval_semantic_degradation_curve/summarize_ifeval_semantic_degradation_curve.py
```

## GSM8K SOTA Boundary Ablation

Purpose: isolate the contribution of the math boundary head and delimiter cue.

Code:

```text
llada/experiments/semantic_boundary_indep/20260519_gsm8k_sota_boundary_ablation/run_gsm8k_sota_boundary_ablation.sh
llada/experiments/semantic_boundary_indep/20260519_gsm8k_sota_boundary_ablation/summarize_gsm8k_sota_boundary_ablation.py
```

Paper artifacts:

```text
paper_artifacts_20260521/gsm8k_boundary_ablation/
```

## HumanEval Boundary Case Study

Concrete AdaBlock-wrong / SemBlock-correct example:

```text
paper_artifacts_20260521/humaneval_boundary_case_study/
```

