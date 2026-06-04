# Checkpoints and Large Artifacts

Large checkpoint and generated data files are not committed to GitHub. The
paper-facing runs used the following local checkpoint paths.

## Boundary Heads

Natural language / GUM head:

```text
llada/checkpoints/gum_direct_20260413/boundary_head_best.pt
```

Math head:

```text
llada/checkpoints/math_external_aqua_only_head_20260428/boundary_head_last.pt
```

Code head:

```text
llada/checkpoints/nonhe_aug_failuretargeted_20260511/boundary_head_step_000250.pt
```

## Base Models

LLaDA-Instruct:

```text
/home/nvme03/workspace/models/GSAI-ML/LLaDA-8B-Instruct
```

LLaDA-1.5:

```text
GSAI-ML/LLaDA-1.5
```

## Published Compact Artifacts

Small result summaries and paper tables are included under:

```text
paper_artifacts_20260521/
```

Full generated caches, traces, raw train/valid JSONL data, and `.pt`
checkpoint files should be stored separately with Git LFS, a GitHub release, or
a model/data hosting service.

