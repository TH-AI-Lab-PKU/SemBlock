# SemBlock

SemBlock is the release repository for semantic-boundary block scheduling in
diffusion LLM inference. It contains the usable code paths for:

- data construction and token-level semantic-boundary labeling;
- training the three task-specialized boundary heads: natural language/GUM,
  math, and code;
- SOTA evaluation on IFEval, GSM8K, MATH, and HumanEval;
- boundary-overlap and boundary-quality ablations;
- post-processing scripts used to summarize ablation and SOTA results.

The repository is intentionally code-first. Large model weights, full raw
datasets, generated caches, traces, and `.pt` checkpoints are not committed to
GitHub. See `docs/CHECKPOINTS_AND_ARTIFACTS.md` for the checkpoint paths used in
the paper-facing runs.

## Structure

```text
llada/
  eval_llada_adablock.py          # main LLaDA evaluation entry
  eval_llada_semantic.py          # semantic-boundary code-head evaluation
  generate_semantic.py            # semantic scheduler generation runtime
  semantic_boundary.py            # boundary head loading/inference
  semantic_scheduler.py           # boundary-aware block scheduler
  semantic_runtime_*.py           # runtime variants used in ablations
  train_boundary_segmenter.py     # boundary-head training
  build_*                         # data construction and augmentation scripts
  run_*                           # SOTA/ablation launchers
  experiments/semantic_boundary_indep/
    20260516_boundary_overlap*/   # boundary-overlap experiment code/results
    20260518_ifeval_*             # IFEval naturalness/degradation ablations
    20260519_gsm8k_*              # GSM8K boundary ablation
paper_artifacts_20260521/
  SOTA_RESULTS.md
  SOTA_HYPERPARAMETERS.md
  boundary_overlap_with_adablock/
  gsm8k_boundary_ablation/
  humaneval_boundary_case_study/
tests/
```

## Environment

```bash
conda env create -f environment.yml
conda activate semb
pip install -r requirements.txt
```

For code execution benchmarks such as HumanEval, export:

```bash
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
```

## Reproducibility Guide

The end-to-end paper workflow is documented in:

- `docs/DATA_AND_LABELS.md`
- `docs/TRAINING.md`
- `docs/EVALUATION.md`
- `docs/ABLATIONS_AND_POSTPROCESSING.md`
- `docs/CHECKPOINTS_AND_ARTIFACTS.md`

The compact paper-facing result summaries are in:

- `paper_artifacts_20260521/SOTA_RESULTS.md`
- `paper_artifacts_20260521/SOTA_HYPERPARAMETERS.md`

## Quick Sanity Checks

```bash
python -m py_compile \
  llada/train_boundary_segmenter.py \
  llada/eval_llada_adablock.py \
  llada/eval_llada_semantic.py \
  llada/compare_boundary_traces.py \
  llada/rescore_hendrycks_math_cache.py

python -m unittest tests.test_compare_boundary_traces tests.test_semantic_scheduler
```

Full evaluation requires local access to LLaDA checkpoints and the task-specific
boundary-head checkpoints listed in `docs/CHECKPOINTS_AND_ARTIFACTS.md`.

