# Training Boundary Heads

The main training entry point is:

```text
llada/train_boundary_segmenter.py
```

It trains a lightweight boundary head on top of tokenized segment examples. The
same training entry is used for the three task lines, with different processed
JSONL data.

## Natural Language / GUM Head

Input data:

```text
llada/data/semantic_boundary/processed/gum/train.jsonl
llada/data/semantic_boundary/processed/gum/valid.jsonl
```

SOTA checkpoint used in evaluation:

```text
checkpoints/gum_direct_20260413/boundary_head_best.pt
```

## Math Head

Input data: processed math step-boundary examples built by
`build_task_conditioned_phase_boundary_jsonl.py` and related math utilities.

SOTA checkpoint used in GSM8K/MATH:

```text
checkpoints/math_external_aqua_only_head_20260428/boundary_head_last.pt
```

## Code Head

Input data: non-HumanEval code examples and targeted failure augmentation.

Key data builders:

```text
llada/build_leetcode_humaneval_bridge_jsonl.py
llada/build_targeted_nonhe_aug.py
```

SOTA checkpoint used in HumanEval:

```text
checkpoints/nonhe_aug_failuretargeted_20260511/boundary_head_step_000250.pt
```

The checkpoint validation metrics were:

```text
valid_boundary_f1=0.47423978510419507
valid_transition_f1=0.8976220275344181
```

## Notes

The `.pt` files are not committed to this GitHub repository because each one is
large enough that it should be stored with Git LFS, a GitHub release, or a model
hub. See `docs/CHECKPOINTS_AND_ARTIFACTS.md`.

