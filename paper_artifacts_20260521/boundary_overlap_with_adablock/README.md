# Boundary Overlap with AdaBlock

这个文件夹整理“我们的语义边界切分与 AdaBlock 边界重合度并不高”的实验材料，供论文写作使用。

## 核心问题

AdaBlock 主要依赖 delimiter / 交叉熵式的局部信号来确定 block 边界。我们的边界来自 task-specific semantic heads 或 semantic hybrid scheduler。这个实验比较两者在同一批样本上的边界集合重合程度，用来说明：

- 我们的方法不是简单复现 AdaBlock 的边界；
- 不同任务上的 semantic head 会产生不同于 AdaBlock 的 boundary schedule；
- 后续 GSM8K ablation 中 hybrid 最强，说明这种差异不仅存在，而且会影响生成性能。

## 论文用结果表

| Task | Semantic head | Target samples | Matched samples | Semantic boundaries | AdaBlock boundaries | Exact overlap | Jaccard |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HumanEval | Code head | 17 | 17 | 336 | 500 | 4.76 | 1.95 |
| IFEval | GUM semantic head | 55 | 55 | 881 | 1373 | 25.88 | 11.25 |
| GSM8K | Math head | 132 | 132 | 3323 | 3088 | 65.45 | 51.35 |
| MATH | Math head | 500 | 351 | 10781 | 10941 | 82.39 | 69.19 |

数值单位为百分比。`Exact overlap` 是 semantic boundaries 中被 AdaBlock 精确命中的比例；`Jaccard` 是两个边界集合的交并比。

## 主要结论

HumanEval 与 IFEval 的重合度很低，说明 code head / GUM semantic head 与 AdaBlock 的 delimiter-style boundary 明显不同。GSM8K 和 MATH 的重合度更高，这是合理的，因为数学推理中换行、公式步骤和答案结构更容易同时触发 semantic head 与 delimiter cue；但它们仍然不是完全一致的边界，尤其 GSM8K 的 Jaccard 只有 51.35%，MATH 的 Jaccard 也不是 100%。

论文中建议表述为：

> The learned semantic boundaries are not a simple reproduction of AdaBlock boundaries. On HumanEval and IFEval, the Jaccard overlap is only 1.95% and 11.25%, respectively. Even on math-heavy tasks where delimiter cues naturally align with reasoning steps, the boundary sets remain non-identical, with Jaccard overlaps of 51.35% on GSM8K and 69.19% on MATH.

## 文件说明

- `experiment_protocol.md`: 具体实验过程、任务/head 对应关系、指标定义。
- `results.md`: 结果解读和论文论证方式。
- `paper_writing_notes.md`: 可直接写进论文的中英文段落。
- `table_boundary_overlap_with_adablock.tex`: LaTeX 表格。
- `results_percent.csv`: 百分比格式结果表。
- `raw_tenth_overlap_summary.md`: 原始 markdown summary。
- `raw_tenth_overlap_summary.csv`: 原始 csv summary。

## 原始实验目录

```text
llada/experiments/semantic_boundary_indep/20260516_boundary_overlap_tenth_corresponding/
```

