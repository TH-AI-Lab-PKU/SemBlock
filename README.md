# SemBlock

SemBlock is the release repository for semantic-boundary block scheduling in
diffusion LLM inference. The main things you run are:

- `llada/train_boundary_segmenter.py` to train a boundary head.
- `llada/eval_llada_adablock.py` to evaluate natural-language/math tasks.
- `llada/eval_llada_semantic.py` to evaluate code tasks such as HumanEval.

Large model weights, raw/generated datasets, caches, traces, and `.pt`
checkpoints are not committed. Before running, place the LLaDA model and the
boundary-head checkpoints on your machine.

## Paper

**SemBlock: Semantic Boundary Dynamic Blocks for Diffusion LLMs**

- Authors: Xinrui Song, Zhuoran Wang, Mingju Gao, Hao Tang
- arXiv: [2606.04964](https://arxiv.org/abs/2606.04964)
- PDF: [arxiv.org/pdf/2606.04964](https://arxiv.org/pdf/2606.04964)
- DOI: [10.48550/arXiv.2606.04964](https://doi.org/10.48550/arXiv.2606.04964)
- Submitted: June 3, 2026

```bibtex
@misc{song2026semblock,
  title={SemBlock: Semantic Boundary Dynamic Blocks for Diffusion LLMs},
  author={Song, Xinrui and Wang, Zhuoran and Gao, Mingju and Tang, Hao},
  year={2026},
  eprint={2606.04964},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  doi={10.48550/arXiv.2606.04964}
}
```

## Setup

Create the environment:

```bash
conda env create -f environment.yml
conda activate semb
pip install -r requirements.txt
```

Use `llada/` as the working directory for the commands below:

```bash
cd llada
export PYTHONPATH="$PWD"
```

Set local paths. Change these to match your machine:

```bash
export MODEL_PATH=/path/to/LLaDA-8B-Instruct
export GUM_HEAD=checkpoints/gum_direct_20260413/boundary_head_best.pt
export MATH_HEAD=checkpoints/math_external_aqua_only_head_20260428/boundary_head_last.pt
export CODE_HEAD=checkpoints/nonhe_aug_failuretargeted_20260511/boundary_head_step_000250.pt
```

For HumanEval/code execution:

```bash
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
```

Optional quick check:

```bash
python -m py_compile \
  train_boundary_segmenter.py \
  eval_llada_adablock.py \
  eval_llada_semantic.py \
  compare_boundary_traces.py \
  rescore_hendrycks_math_cache.py

cd ..
python -m unittest tests.test_compare_boundary_traces tests.test_semantic_scheduler
cd llada
```

## Data

Processed JSONL files are not included in this repo. Build or place them before
training.

The training script accepts rows with `segments`, `input_ids + boundary_labels`,
or `input_ids + boundary_positions`.

Useful builders:

```bash
echo "GUM / natural-language data"
python prepare_boundary_corpora.py --help

echo "Math/task-conditioned phase-boundary data"
python build_task_conditioned_phase_boundary_jsonl.py --help

echo "Code data from LeetCode-style sources"
python build_leetcode_humaneval_bridge_jsonl.py --help

echo "Code targeted augmentation"
python build_targeted_nonhe_aug.py --help
```

Expected locations used by the commands below:

```text
data/semantic_boundary/processed/gum/train.jsonl
data/semantic_boundary/processed/gum/valid.jsonl
data/semantic_boundary/processed/combined/train.jsonl
data/semantic_boundary/processed/combined/valid.jsonl
data/semantic_boundary/processed/nonhumaneval_code_leetcode_aug_failuretargeted_20260511/train.jsonl
data/semantic_boundary/processed/nonhumaneval_code_leetcode_aug_failuretargeted_20260511/valid.jsonl
```

The common label convention is:

```text
last token of the segment -> boundary label 1
all other tokens          -> boundary label 0
```

## Train

After data is ready, train a boundary head with
`train_boundary_segmenter.py`:

```bash
python train_boundary_segmenter.py \
  --model_path "$MODEL_PATH" \
  --train_data data/semantic_boundary/processed/gum/train.jsonl \
  --valid_data data/semantic_boundary/processed/gum/valid.jsonl \
  --output_dir checkpoints/my_gum_head \
  --epochs 3 \
  --batch_size 2 \
  --learning_rate 1e-4 \
  --max_length 1024
```

The output is written to:

```text
checkpoints/my_gum_head/boundary_head_last.pt
checkpoints/my_gum_head/boundary_head_best.pt
checkpoints/my_gum_head/metrics.json
```

For a quick smoke run, add:

```bash
--max_train_steps 10
```

### Train the Three Heads

Natural-language / GUM head:

```bash
python train_boundary_segmenter.py \
  --model_path "$MODEL_PATH" \
  --train_data data/semantic_boundary/processed/gum/train.jsonl \
  --valid_data data/semantic_boundary/processed/gum/valid.jsonl \
  --output_dir checkpoints/gum_direct_20260413 \
  --epochs 3 \
  --batch_size 2
```

Math head:

```bash
python train_boundary_segmenter.py \
  --model_path "$MODEL_PATH" \
  --train_data data/semantic_boundary/processed/combined/train.jsonl \
  --valid_data data/semantic_boundary/processed/combined/valid.jsonl \
  --output_dir checkpoints/math_external_aqua_only_head_20260428 \
  --epochs 3 \
  --batch_size 2
```

Code / HumanEval head:

```bash
python train_boundary_segmenter.py \
  --model_path "$MODEL_PATH" \
  --train_data data/semantic_boundary/processed/nonhumaneval_code_leetcode_aug_failuretargeted_20260511/train.jsonl \
  --valid_data data/semantic_boundary/processed/nonhumaneval_code_leetcode_aug_failuretargeted_20260511/valid.jsonl \
  --output_dir checkpoints/nonhe_aug_failuretargeted_20260511 \
  --epochs 3 \
  --batch_size 2 \
  --checkpoint_steps 250
```

The code-head checkpoint used by the paper is:

```text
checkpoints/nonhe_aug_failuretargeted_20260511/boundary_head_step_000250.pt
```

## Evaluate

Run evaluation from `llada/`. The result JSONs go to `--output_path`; generated
answers/caches go to the `save_dir=...` inside `--model_args`.

### IFEval

```bash
python eval_llada_adablock.py \
  --model llada_dist \
  --tasks ifeval \
  --model_args "model_path=${MODEL_PATH},block_strategy=semantic_head,block_length=32,steps=16,gen_length=512,threshold=0.9,show_speed=True,use_cache=True,boundary_prior_path=${GUM_HEAD},boundary_prior_threshold=0.75,save_dir=runs/ifeval_semblock_b32" \
  --batch_size 1 \
  --output_path eval_results/ifeval_semblock_b32 \
  --log_samples
```

### GSM8K

```bash
python eval_llada_adablock.py \
  --model llada_dist \
  --tasks gsm8k \
  --num_fewshot 5 \
  --model_args "model_path=${MODEL_PATH},block_strategy=semantic_hybrid,block_length=32,steps=16,gen_length=512,threshold=0.9,task_type=math,show_speed=True,use_cache=True,delimiter_ids=198,delimiter_threshold=0.3,gsm8k_landing_control=true,boundary_prior_path=${MATH_HEAD},boundary_prior_threshold=0.60,boundary_prior_weight=0.70,semantic_min_block_length=8,semantic_selection_mode=max_score_above_threshold,save_dir=runs/gsm8k_semblock_b32" \
  --batch_size 1 \
  --output_path eval_results/gsm8k_semblock_b32 \
  --log_samples
```

### MATH

First generate/cache model outputs:

```bash
python eval_llada_adablock.py \
  --model llada_dist \
  --tasks hendrycks_math \
  --confirm_run_unsafe_code \
  --model_args "model_path=${MODEL_PATH},block_strategy=semantic_hybrid,block_length=32,steps=16,gen_length=512,threshold=0.9,task_type=math,show_speed=True,use_cache=True,delimiter_ids=198,delimiter_threshold=0.3,gsm8k_landing_control=false,boundary_prior_path=${MATH_HEAD},boundary_prior_threshold=0.60,boundary_prior_weight=0.70,semantic_min_block_length=8,semantic_selection_mode=max_score_above_threshold,save_dir=runs/math_semblock_b32" \
  --batch_size 1 \
  --output_path eval_results/math_semblock_b32 \
  --log_samples
```

Then run the paper-style boxed-answer rescore:

```bash
python rescore_hendrycks_math_cache.py \
  --cache-jsonl runs/math_semblock_b32/rank_0.jsonl \
  --output-json eval_results/math_semblock_b32_offline_rescore.json
```

### HumanEval

```bash
python eval_llada_semantic.py \
  --model llada_semantic \
  --tasks llada_humaneval_subset \
  --include_path lm_eval_tasks \
  --num_fewshot 0 \
  --confirm_run_unsafe_code \
  --model_args "model_path=${MODEL_PATH},gen_length=512,steps=32,block_length=32,threshold=0.9,boundary_head_path=${CODE_HEAD},boundary_threshold=0.5,boundary_window_ratio=0.125,scheduler_variant=failure_case_router,phase_entropy_gate=0.8,transition_weight=0.0,runtime_mode=boundary_only,use_cache=False,show_speed=True,save_dir=runs/humaneval_semblock_b32/generations,trace_dir=runs/humaneval_semblock_b32/traces,phase_aware_transfer=False,boundary_guard=False,syntax_aware_landing=True,commit_reopen_tokens=8" \
  --output_path eval_results/humaneval_semblock_b32 \
  --log_samples
```

To run one HumanEval shard with the included paper runner instead:

```bash
bash run_humaneval_code_sota_exact_20260518.sh 0 32 0
```

That script has paper-machine paths hard-coded near the top. Edit
`REPO_DIR`, `ACCELERATE_BIN`, `MODEL_PATH`, and `BOUNDARY_HEAD` before using it
on a new machine.

### Paper Runner Scripts

These scripts reproduce the paper-facing runs but contain absolute paths from
the original machines. Use the direct commands above first; use these only
after editing their path variables.

```bash
bash run_ifeval_gsm8k_queue_20260516.sh 0 32
bash run_hendrycks_math_b016_b064_20260516.sh 0 32
bash run_humaneval_code_sota_exact_20260518.sh 0 32 0
```

Argument meanings:

```text
0   -> NPU/GPU device id
32  -> block length
0   -> HumanEval shard id
```

## Outputs and Artifacts

Compact paper result summaries are committed here:

```text
paper_artifacts_20260521/SOTA_RESULTS.md
paper_artifacts_20260521/SOTA_HYPERPARAMETERS.md
paper_artifacts_20260521/boundary_overlap_with_adablock/
paper_artifacts_20260521/gsm8k_boundary_ablation/
paper_artifacts_20260521/humaneval_boundary_case_study/
```

Main implementation files:

```text
llada/train_boundary_segmenter.py
llada/eval_llada_adablock.py
llada/eval_llada_semantic.py
llada/generate_semantic.py
llada/semantic_boundary.py
llada/semantic_scheduler.py
llada/semantic_runtime_*.py
```
