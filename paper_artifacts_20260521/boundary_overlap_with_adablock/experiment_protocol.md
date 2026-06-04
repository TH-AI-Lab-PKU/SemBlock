# Experiment Protocol

## 1. Goal

这组实验用于比较我们的 semantic boundary scheduler 与 AdaBlock scheduler 在相同样本上的边界重合程度。

实验回答的问题是：我们的 semantic head / semantic hybrid 是否只是复现了 AdaBlock 的边界？如果重合度较低，则说明我们的边界学习到了不同的任务语义结构。

## 2. Task-Head Correspondence

按照之前确认的对应关系：

| Task | Semantic boundary source | AdaBlock baseline |
| --- | --- | --- |
| HumanEval | Code head | AdaBlock delimiter-style boundary |
| IFEval | GUM semantic head | AdaBlock delimiter-style boundary |
| GSM8K | Math head / semantic hybrid | AdaBlock delimiter-style boundary |
| MATH | Math head / semantic hybrid | AdaBlock delimiter-style boundary |

这里不是所有 head 与所有任务相乘，而是按任务对应关系比较。

## 3. Sample Size

这组采用轻量统计版本：

| Task | Target samples | Notes |
| --- | ---: | --- |
| HumanEval | 17 | Roughly one tenth of HumanEval |
| IFEval | 55 | Roughly one tenth of IFEval |
| GSM8K | 132 | Roughly one tenth of GSM8K |
| MATH | 500 | Later re-run target for MATH |

MATH 的最终 matched samples 是 351/500，因此表中标为 partial 结果。论文中如果引用 MATH，应写清楚它是 partial matched result，或只把 MATH 当作补充观察。

## 4. Running Script

原始脚本：

```text
llada/experiments/semantic_boundary_indep/20260516_boundary_overlap_tenth_corresponding/run_tenth_corresponding_overlap.sh
```

MATH 500 条补跑脚本：

```text
llada/experiments/semantic_boundary_indep/20260516_boundary_overlap_tenth_corresponding/run_math500_adablock_and_compare.sh
```

汇总脚本：

```text
llada/experiments/semantic_boundary_indep/20260516_boundary_overlap_tenth_corresponding/summarize_tenth_overlap.py
```

原始输出：

```text
llada/experiments/semantic_boundary_indep/20260516_boundary_overlap_tenth_corresponding/tenth_overlap_summary.md
llada/experiments/semantic_boundary_indep/20260516_boundary_overlap_tenth_corresponding/tenth_overlap_summary.csv
llada/experiments/semantic_boundary_indep/20260516_boundary_overlap_tenth_corresponding/overlap/
```

## 5. Metric Definitions

### Exact overlap

`Exact overlap` 表示 semantic boundaries 中有多少比例与 AdaBlock boundary 在同一个位置精确重合。

在原始 JSON 中对应：

```text
task_summaries[0].exact.semantic_to_adablock_precision
```

### Jaccard

`Jaccard` 表示 semantic boundary set 与 AdaBlock boundary set 的交并比：

```text
|semantic boundaries ∩ AdaBlock boundaries| / |semantic boundaries ∪ AdaBlock boundaries|
```

Jaccard 越低，说明两个 scheduler 的整体边界集合越不同。

## 6. Results

| Task | Semantic head | Target samples | Status | Matched samples | Semantic boundaries | AdaBlock boundaries | Exact overlap | Jaccard |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| HumanEval | Code head | 17 | done | 17 | 336 | 500 | 4.76% | 1.95% |
| IFEval | GUM semantic head | 55 | done | 55 | 881 | 1373 | 25.88% | 11.25% |
| GSM8K | Math head | 132 | done | 132 | 3323 | 3088 | 65.45% | 51.35% |
| MATH | Math head | 500 | partial 351/500 | 351 | 10781 | 10941 | 82.39% | 69.19% |

## 7. Caveats

- HumanEval / IFEval / GSM8K 是 roughly one-tenth statistics，不是 full-set overlap。
- MATH 是 target 500、matched 351 的 partial 结果。
- 对于 MATH 和 GSM8K，重合度比自然语言和代码任务更高，这与数学推理中自然换行和步骤边界更清晰有关。论文中不建议声称所有任务都“极低重合”；更准确的说法是：我们的边界并非简单复制 AdaBlock，且在 code/instruction-following tasks 上差异尤其明显。

