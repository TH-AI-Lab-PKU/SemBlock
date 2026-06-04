# Semantic Boundary Prior for LLaDA AdaBlock

This extension adds a learned semantic boundary prior between the first
LLaDA denoising pass and AdaBlock's block-size scheduler.

## What was added

- `semantic_boundary.py`
  - Lightweight boundary head on top of frozen LLaDA hidden states.
  - Utilities for saving and loading boundary checkpoints.
- `train_boundary_segmenter.py`
  - Trains the boundary head from JSONL supervision.
- `generate_adablock.py`
  - Blends delimiter confidence with learned semantic boundary scores.
- `eval_llada_adablock.py`
  - Loads a boundary checkpoint and passes it into AdaBlock generation.

## Training data format

Use JSONL. Each line may use one of the following schemas.

### Option 1: semantic segments

```json
{"source": "gum", "segments": ["This is EDU one.", " This is EDU two."]}
```

Each segment should preserve its exact whitespace and punctuation so that the
tokenization boundary stays aligned.

### Option 2: token ids with dense labels

```json
{"source": "proofnet", "input_ids": [1, 2, 3], "boundary_labels": [0, 1, 1]}
```

Label semantics:

- `boundary_labels[i] == 1` means a semantic boundary occurs after token `i`.

### Option 3: token ids with sparse boundary positions

```json
{"source": "juice", "input_ids": [1, 2, 3, 4], "boundary_positions": [1, 3]}
```

## Suggested corpus mapping

- GUM: EDU or discourse unit segments.
- CodeSearchNet: weak statement or AST chunk segments.
- JuICe: alternating markdown and code cell segments.
- Lean Workbook: theorem, lemma, tactic block, or statement segments.
- ProofNet: gold proof-step segments.

## Train

```bash
python train_boundary_segmenter.py \
  --model_path GSAI-ML/LLaDA-8B-Instruct \
  --train_data /path/to/boundary_train.jsonl \
  --valid_data /path/to/boundary_valid.jsonl \
  --output_dir /path/to/boundary_ckpt \
  --max_length 1024 \
  --batch_size 2 \
  --epochs 3 \
  --learning_rate 1e-4 \
  --max_noise_ratio 0.5
```

The script freezes LLaDA and trains only the boundary head. During training it
randomly masks part of the input to better match diffusion denoising conditions.
By default the final token in each segment is also labeled as a boundary. Use
`--exclude_terminal_boundary` only if your data treats terminal boundaries as
out of scope.

## Use inside AdaBlock evaluation

Pass the new model args through `lm-eval`:

```bash
accelerate launch eval_llada_adablock.py \
  --model llada_dist \
  --model_args pretrained=GSAI-ML/LLaDA-8B-Instruct,threshold=0.9,delimiter_threshold=0.3,boundary_prior_path=/path/to/boundary_ckpt/boundary_head_best.pt,boundary_prior_weight=0.7,boundary_prior_threshold=0.55
```

Key knobs:

- `boundary_prior_path`: checkpoint from `train_boundary_segmenter.py`
- `boundary_prior_weight`: blend weight for semantic prior vs. delimiter confidence
- `boundary_prior_threshold`: accept or reject the blended boundary proposal
- `boundary_window_ratio`: size of the look-ahead window used by the scheduler

## Practical recommendation

Start by mixing all corpora into one normalized JSONL and train a single head.
Then compare:

1. delimiter-only AdaBlock
2. semantic prior only with a high `boundary_prior_weight`
3. blended semantic prior plus delimiter confidence

This usually makes it easier to see whether the segmenter is helping block
placement itself or only helping when punctuation is already present.
