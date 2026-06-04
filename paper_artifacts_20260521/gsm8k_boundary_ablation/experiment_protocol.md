# Experiment Protocol

## 1. Motivation

之前的 boundary overlap 实验说明：我们的 semantic boundary 和 AdaBlock-style boundary 并不是同一种划分方式。  
这组 GSM8K 消融进一步回答一个性能问题：在同一个 SOTA decoding setup 下，是否必须同时使用 learned task-semantic boundary 和 natural-language delimiter cue，才能得到最好效果。

GSM8K 被选作主消融任务，原因是：

- 它对应 math head，是当前 SOTA 设置中最清晰的 task-semantic boundary 场景。
- 与 IFEval random/jitter degradation 相比，这组消融不改变任务、不改变模型、不改变大部分 decoding 参数，只移除 boundary scheduler 的一个组成部分，因此更适合写入论文。
- 它能区分“更多 compute / 更细 block”和“boundary quality”两个解释。

## 2. Controlled Setup

所有 run 都使用 GSM8K B=32 SOTA decoding setup，除 boundary signal 外尽量保持一致。

固定配置：

```text
model_path=/home/nvme03/workspace/models/GSAI-ML/LLaDA-8B-Instruct
task=gsm8k
num_fewshot=5
limit=300
batch_size=1
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

Math boundary head：

```text
/home/nvme01/workspace/AdaBlock-dLLM-main/llada/checkpoints/math_external_aqua_only_head_20260428/boundary_head_last.pt
```

Semantic hybrid hyperparameters：

```text
boundary_prior_threshold=0.60
semantic_min_block_length=8
semantic_selection_mode=max_score_above_threshold
```

## 3. Variants

### 3.1 `sota_hybrid`

已有 SOTA reference，不重新跑。

Boundary signal：

```text
math head + natural-language newline delimiter
```

关键配置：

```text
block_strategy=semantic_hybrid
boundary_prior_weight=0.7
delimiter_threshold=0.3
```

原始结果文件：

```text
/home/nvme01/workspace/AdaBlock-dLLM-main/llada/eval_results_math_semantic/aqua_gsm8k_confirm_l300_20260502/gsm8k/semantic_hybrid_b32_cache_on_limit300_thr0p60_minb8_selmax_score_above_threshold_mix0p70_landtrue/__home__nvme03__workspace__models__GSAI-ML__LLaDA-8B-Instruct/results_2026-05-03T00-19-41.222408.json
```

原始日志：

```text
/home/nvme01/workspace/AdaBlock-dLLM-main/llada/logs/math_semantic/aqua_gsm8k_confirm_l300_20260502/gsm8k/semantic_hybrid_b32_cache_on_limit300_thr0p60_minb8_selmax_score_above_threshold_mix0p70_landtrue.log
```

### 3.2 `delimiter_only_adablock`

目的：移除 learned math boundary head，只保留自然语言 delimiter cue。

关键配置：

```text
block_strategy=adablock
delimiter_ids=198
delimiter_threshold=0.3
```

结果文件：

```text
llada/experiments/semantic_boundary_indep/20260519_gsm8k_sota_boundary_ablation/results/delimiter_only_adablock/__home__nvme03__workspace__models__GSAI-ML__LLaDA-8B-Instruct/results_2026-05-19T21-50-43.928139.json
```

日志：

```text
llada/experiments/semantic_boundary_indep/20260519_gsm8k_sota_boundary_ablation/logs/delimiter_only_adablock.log
```

### 3.3 `math_head_only`

目的：移除 delimiter confidence 对 hybrid score 的影响，只使用 learned math boundary head。

关键配置：

```text
block_strategy=semantic_hybrid
boundary_prior_threshold=0.60
boundary_prior_weight=1.0
semantic_min_block_length=8
semantic_selection_mode=max_score_above_threshold
delimiter_ids=198
delimiter_threshold=0.3
```

说明：这里保留 `delimiter_ids` 和 `delimiter_threshold` 在参数中，是为了保持基础配置一致；真正使 delimiter confidence 不参与 hybrid score 的是 `boundary_prior_weight=1.0`。

结果文件：

```text
llada/experiments/semantic_boundary_indep/20260519_gsm8k_sota_boundary_ablation/results/math_head_only/__home__nvme03__workspace__models__GSAI-ML__LLaDA-8B-Instruct/results_2026-05-20T01-21-03.253589.json
```

日志：

```text
llada/experiments/semantic_boundary_indep/20260519_gsm8k_sota_boundary_ablation/logs/math_head_only.log
```

## 4. Running Procedure

使用脚本：

```text
llada/experiments/semantic_boundary_indep/20260519_gsm8k_sota_boundary_ablation/run_gsm8k_sota_boundary_ablation.sh
```

脚本行为：

- 读取已有 SOTA reference，不重跑 SOTA。
- 运行 `delimiter_only_adablock`。
- 运行 `math_head_only`。
- 每个 variant 完成后调用 summarizer 汇总结果。
- 默认只在 `ALLOWED_GPUS=0,1` 中选择 GPU，并避免当时不允许使用的 GPU 2/3/4/5。

汇总脚本：

```text
llada/experiments/semantic_boundary_indep/20260519_gsm8k_sota_boundary_ablation/summarize_gsm8k_sota_boundary_ablation.py
```

汇总输出：

```text
llada/experiments/semantic_boundary_indep/20260519_gsm8k_sota_boundary_ablation/gsm8k_sota_boundary_ablation_summary.md
llada/experiments/semantic_boundary_indep/20260519_gsm8k_sota_boundary_ablation/gsm8k_sota_boundary_ablation_summary.csv
```

## 5. Caveats

- 这组结果使用 `limit=300`，是为了快速消融并与已有 limit-300 SOTA reference 对齐。论文中如果作为正式主表，最好标注为 300-sample ablation 或后续补 full-set。
- `lm_eval` 日志中会提示 `--limit SHOULD ONLY BE USED FOR TESTING`，这不影响 ablation 的相对比较，但写论文时需要避免把这组 300-sample 数字混成 full-set benchmark。
- 这组实验最强的论证点不是“自然语言边界单独最好”，而是“task semantic boundary 与 natural-language delimiter cue 互补，hybrid 最好”。

