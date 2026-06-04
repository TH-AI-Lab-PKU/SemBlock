# Paper Writing Notes

## 中文版本

> 为验证我们的语义边界并非简单复现 AdaBlock 的边界，我们在 HumanEval、IFEval、GSM8K 和 MATH 上统计了两类边界的重合度。结果显示，在 HumanEval 和 IFEval 上，semantic boundary 与 AdaBlock boundary 的 Jaccard 仅为 1.95% 和 11.25%，说明 code head 与 GUM semantic head 学到的边界与 delimiter-style 的 AdaBlock 边界明显不同。在 GSM8K 和 MATH 上，由于数学解答天然包含换行、推理步骤和公式结构，两者重合度相对更高，但仍非完全一致，Jaccard 分别为 51.35% 和 69.19%。这说明我们的 semantic scheduler 并不是 AdaBlock 的简单复制，而是在不同任务中捕获了额外的任务语义结构。

更短版本：

> Boundary-overlap results show that our learned semantic boundaries are not a direct reproduction of AdaBlock boundaries. The Jaccard overlap is only 1.95% on HumanEval and 11.25% on IFEval, and remains imperfect on math tasks despite their stronger delimiter-step alignment.

## English Version

Full paragraph:

> To verify that our semantic scheduler does not simply reproduce AdaBlock boundaries, we measure boundary overlap on HumanEval, IFEval, GSM8K, and MATH. The overlap is very low on HumanEval and IFEval, with Jaccard scores of 1.95% and 11.25%, respectively, indicating that the code and GUM semantic heads identify substantially different boundary structures from AdaBlock. On GSM8K and MATH, the overlap is higher because math solutions naturally contain line breaks, reasoning steps, and equation boundaries that can be captured by both methods. Nevertheless, the boundary sets remain non-identical, with Jaccard scores of 51.35% and 69.19%. These results suggest that our scheduler captures task-semantic structure beyond delimiter-style AdaBlock boundaries.

Short paragraph:

> Boundary-overlap analysis shows that the learned semantic scheduler is not a direct copy of AdaBlock. The Jaccard overlap is only 1.95% on HumanEval and 11.25% on IFEval. Even on math tasks, where delimiter cues naturally align with reasoning steps, the overlap remains imperfect.

## Recommended Claim Strength

Recommended:

```text
Our semantic boundaries are not a simple reproduction of AdaBlock boundaries.
```

Recommended:

```text
The two schedulers capture different and partially complementary boundary structures.
```

Use carefully:

```text
The overlap is low on all tasks.
```

This is too strong because GSM8K and MATH have moderate-to-high overlap. A safer statement is:

```text
The overlap is low on code and instruction-following tasks, and remains non-identical on math tasks.
```

## Suggested Caption

> Boundary overlap between our semantic scheduler and AdaBlock. The low Jaccard scores on HumanEval and IFEval show that learned semantic heads do not simply reproduce AdaBlock boundaries. Math tasks show higher overlap due to natural step and delimiter alignment, but the boundary sets remain non-identical.

