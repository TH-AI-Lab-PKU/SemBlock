# SOTA Hyperparameters

Date: 2026-05-22

This file records the paper-facing hyperparameter settings for the SOTA cells
that were found in the local experiment logs, scripts, and archived result
folders.

## Summary Table

| Task | Model | Main result | Boundary source | B0 / block | Key decoding hyperparameters | Evidence |
|---|---|---:|---|---:|---|---|
| GSM8K | LLaDA-Instruct | 82.60 | Math head + semantic hybrid | 32 | `steps=16`, `gen_length=512`, `threshold=0.9`, `use_cache=True`, `num_fewshot=5` | `gsm8k_boundary_ablation/experiment_protocol.md` |
| IFEval | LLaDA-Instruct | 56.00 | GUM semantic head | 16 | `steps=16`, `gen_length=512`, `threshold=0.9`, `use_cache=True`, full 541 prompts | `run_ifeval_gsm8k_queue_20260516.sh` |
| IFEval | LLaDA-Instruct | 56.93 | GUM semantic head | 32 | `steps=16`, `gen_length=512`, `threshold=0.9`, `use_cache=True`, full 541 prompts | `full_ifeval_threshold_best_20260415` |
| HumanEval | LLaDA-Instruct | 46.95 pass@1 | Code head / generalized router v3 | 32 | `steps=32`, `gen_length=512`, `threshold=0.9`, `use_cache=False`, zero-shot | `SOTA_ARCHIVE_20260512_generalized_v3_4695` |
| HumanEval | LLaDA-1.5 | 47.56 pass@1 | OpenCompass confidence config | 16 | `gen_steps=512`, `gen_length=512`, `diff_confidence_eos_eot_inf=True`, zero-shot | `opencompass_llada_1p5_humaneval_b16_confidence.py` |
| HumanEval | LLaDA-1.5 | 49.39 pass@1 | OpenCompass confidence config | 32 | Same as B16/B64 pattern with `gen_blocksize=32`; raw summary preserved | `results_raw/humaneval_llada15_b32_summary.csv` |
| HumanEval | LLaDA-1.5 | 50.00 pass@1 | OpenCompass confidence config | 64 | `gen_steps=512`, `gen_length=512`, `diff_confidence_eos_eot_inf=True`, zero-shot | `opencompass_llada_1p5_humaneval_b64_confidence.py` |
| MATH | LLaDA-Instruct | 37.80 | Math head + semantic hybrid, offline rescore | 32 | `steps=16`, `gen_length=512`, `threshold=0.9`, `use_cache=True`, zero-shot | `offline_rescore_b32_20260516_check.json` |

## GSM8K: LLaDA-Instruct, B0=32

Result:

```text
GSM8K strict exact match: 82.60
```

Core model/evaluation settings:

```text
model_path=/home/nvme03/workspace/models/GSAI-ML/LLaDA-8B-Instruct
task=gsm8k
num_fewshot=5
batch_size=1
block_strategy=semantic_hybrid
block_length=32
steps=16
gen_length=512
threshold=0.9
task_type=math
use_cache=True
delimiter_ids=198
delimiter_threshold=0.3
gsm8k_landing_control=true
```

Boundary settings:

```text
boundary_prior_path=/home/nvme01/workspace/AdaBlock-dLLM-main/llada/checkpoints/math_external_aqua_only_head_20260428/boundary_head_last.pt
boundary_prior_threshold=0.60
boundary_prior_weight=0.70
semantic_min_block_length=8
semantic_selection_mode=max_score_above_threshold
```

Primary record:

```text
/home/nvme01/workspace/AdaBlock-dLLM-main/llada/eval_results_math_semantic/aqua_gsm8k_confirm_l300_20260502/gsm8k/semantic_hybrid_b32_cache_on_limit300_thr0p60_minb8_selmax_score_above_threshold_mix0p70_landtrue/
```

Note: the preserved ablation protocol uses the 300-sample confirmation run as
the reference. Do not mix the ablation number with full-set reporting unless the
paper explicitly labels it as the 300-sample setting.

## IFEval: LLaDA-Instruct, GUM Semantic Head

### B0=32 SOTA Cell

Result:

```text
prompt_level_strict_acc = 0.5600739371534196
paper table value       = 56.93
```

Core model/evaluation settings from the full threshold-best log:

```text
model_path=/home/nvme03/workspace/models/GSAI-ML/LLaDA-8B-Instruct
task=ifeval_local
include_path=/home/nvme01/workspace/AdaBlock-dLLM-main/llada/eval_tasks
limit=541
batch_size=1
block_strategy=semantic_head
block_length=32
steps=16
gen_length=512
threshold=0.9
show_speed=True
use_cache=True
boundary_prior_path=/home/nvme01/workspace/AdaBlock-dLLM-main/llada/checkpoints/gum_direct_20260413/boundary_head_best.pt
boundary_prior_threshold=0.75
```

Primary result:

```text
/home/nvme01/workspace/AdaBlock-dLLM-main/llada/eval_results_ifeval/full_ifeval_threshold_best_20260415/semantic_head_b32_cache_on_limit541_thr0p75/__home__nvme03__workspace__models__GSAI-ML__LLaDA-8B-Instruct/results_2026-04-15T22-47-56.005062.json
```

Primary log:

```text
/home/nvme01/workspace/AdaBlock-dLLM-main/llada/logs/ifeval/full_ifeval_threshold_best_20260415/semantic_head_b32_cache_on_limit541_thr0p75.log
```

### B0=16 and B0=64 Recorded Runs

The later B0=16/B0=64 queue script used the same model, task, cache, and GUM
head, but passed:

```text
boundary_prior_threshold=0.5
```

Source script:

```text
llada/run_ifeval_gsm8k_queue_20260516.sh
```

Reported companion measurements in `SOTA_RESULTS.md`:

```text
B0=16: 56.00
B0=64: 51.20
```

## HumanEval: LLaDA-Instruct, B0=32

This is the missing SOTA cell remembered from the earlier table.

Result:

```text
passed = 77 / 164
pass@1 = 0.4695121951219512
paper table value = 46.95
baseline to beat = 46.3
```

Main lm-eval model args:

```text
model_path=/home/nvme03/workspace/models/GSAI-ML/LLaDA-8B-Instruct
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
show_speed=True
phase_aware_transfer=False
boundary_guard=False
syntax_aware_landing=True
commit_reopen_tokens=8
batch_size=1
num_fewshot=0
```

Boundary-head validation metrics for the checkpoint:

```text
optimizer_step=250
valid_boundary_precision=0.31446383345857337
valid_boundary_recall=0.9640784982935153
valid_boundary_f1=0.47423978510419507
valid_transition_f1=0.8976220275344181
```

Primary archive:

```text
llada/experiments/semantic_boundary_indep/SOTA_ARCHIVE_20260512_generalized_v3_4695/
```

Primary summary:

```text
llada/experiments/semantic_boundary_indep/SOTA_ARCHIVE_20260512_generalized_v3_4695/results/full_humaneval_router_generalized_v3_sharded/merged_summary.json
```

The archive states that HumanEval was only used as benchmark/evaluation data,
not as training data.

## HumanEval: LLaDA-1.5 OpenCompass Cells

Preserved B0=16 and B0=64 configs:

```text
llada/opencompass_llada_1p5_humaneval_b16_confidence.py
llada/opencompass_llada_1p5_humaneval_b64_confidence.py
```

Shared settings:

```text
path=GSAI-ML/LLaDA-1.5
dataset=opencompass/humaneval
prompt=Complete the following python code:\n{prompt}
retriever=ZeroRetriever
batch_size=1
batch_size_=1
max_out_len=1024
gen_length=512
gen_steps=512
diff_confidence_eos_eot_inf=True
diff_logits_eos_inf=False
num_gpus=1
```

B-specific setting:

```text
B0=16: gen_blocksize=16, pass@1=47.56
B0=32: gen_blocksize=32, pass@1=49.39
B0=64: gen_blocksize=64, pass@1=50.00
```

Important caveat: only B0=16 and B0=64 config files are preserved in this
artifact folder. B0=32 is preserved as a raw result summary, and its config is
the same OpenCompass confidence pattern with `gen_blocksize=32`.

## MATH: LLaDA-Instruct, B0=32

Result:

```text
offline rescore exact_match = 0.3776
paper table value           = 37.80
```

Core settings:

```text
model_path=/home/nvme03/workspace/models/GSAI-ML/LLaDA-8B-Instruct
task=hendrycks_math
batch_size=1
block_strategy=semantic_hybrid
block_length=32
steps=16
gen_length=512
threshold=0.9
task_type=math
show_speed=True
use_cache=True
delimiter_ids=198
delimiter_threshold=0.3
gsm8k_landing_control=false
```

Boundary settings:

```text
boundary_prior_path=/home/nvme01/workspace/AdaBlock-dLLM-main/llada/checkpoints/math_external_aqua_only_head_20260428/boundary_head_last.pt
boundary_prior_threshold=0.60
boundary_prior_weight=0.7
semantic_min_block_length=8
semantic_selection_mode=max_score_above_threshold
```

Primary cache and offline rescore:

```text
cache_jsonl=/home/nvme01/workspace/AdaBlock-dLLM-main/llada/resume_cache_math/aqua_hendrycks_math_full_20260512/hendrycks_math/semantic_hybrid_b32_cache_on_full_thr0p60_minb8_selmax_score_above_threshold_mix0p70_dthr0p30_landfalse/rank_0.jsonl
rescore_summary=llada/offline_rescore_b32_20260516_check.json
```

The raw lm-eval exact-match table reported 0.0 for this run, so the paper value
comes from the offline boxed-answer rescore rather than the direct lm-eval
metric.

## Companion B0=16/B0=64 Math-Line Runs

These are recorded measurements from the same math-line setup, but they are not
the primary SOTA cells against the comparison table.

GSM8K B0=16/B0=64 used:

```text
llada/run_ifeval_gsm8k_queue_20260516.sh
block_strategy=semantic_hybrid
boundary_prior_path=/home/nvme01/workspace/AdaBlock-dLLM-main/llada/checkpoints/math_external_aqua_only_head_20260428/boundary_head_last.pt
boundary_prior_threshold=0.60
boundary_prior_weight=0.70
semantic_min_block_length=8
semantic_selection_mode=max_score_above_threshold
delimiter_threshold=0.3
gsm8k_landing_control=true
```

Reported companion measurements in `SOTA_RESULTS.md`:

```text
GSM8K B0=16: 80.36
GSM8K B0=64: 78.77
```

Hendrycks MATH B0=16/B0=64 runs are scripted in:

```text
llada/run_hendrycks_math_b016_b064_20260516.sh
```

They use the same MATH semantic-hybrid settings as B0=32 except for
`block_length`.

## Source Index

| Item | Path |
|---|---|
| SOTA result summary | `paper_artifacts_20260521/SOTA_RESULTS.md` |
| GSM8K ablation and SOTA reference | `paper_artifacts_20260521/gsm8k_boundary_ablation/experiment_protocol.md` |
| IFEval B0=32 log | `/home/nvme01/workspace/AdaBlock-dLLM-main/llada/logs/ifeval/full_ifeval_threshold_best_20260415/semantic_head_b32_cache_on_limit541_thr0p75.log` |
| IFEval B0=32 result | `/home/nvme01/workspace/AdaBlock-dLLM-main/llada/eval_results_ifeval/full_ifeval_threshold_best_20260415/semantic_head_b32_cache_on_limit541_thr0p75/__home__nvme03__workspace__models__GSAI-ML__LLaDA-8B-Instruct/results_2026-04-15T22-47-56.005062.json` |
| HumanEval Instruct SOTA archive | `llada/experiments/semantic_boundary_indep/SOTA_ARCHIVE_20260512_generalized_v3_4695/README.md` |
| HumanEval LLaDA-1.5 B16/B64 runner | `llada/run_opencompass_llada_1p5_humaneval_b016_b064_20260519.sh` |
| MATH offline rescore | `llada/offline_rescore_b32_20260516_check.json` |
| MATH full B32 log | `/home/nvme01/workspace/AdaBlock-dLLM-main/llada/logs/math_semantic/aqua_hendrycks_math_full_20260512/hendrycks_math/semantic_hybrid_b32_cache_on_full_thr0p60_minb8_selmax_score_above_threshold_mix0p70_dthr0p30_landfalse.log` |
