# Paper Writing Notes

## 中文论文思路

这组消融实验的写法建议围绕“边界质量”而不是“单一边界来源”来展开：

> 为验证语义边界质量对生成性能的影响，我们在 GSM8K 上基于 SOTA 的 B=32 设置进行消融。完整模型同时使用 learned math boundary head 与自然语言 newline delimiter cue。我们分别移除 learned math boundary head 和 delimiter confidence，得到 delimiter-only 与 math-head-only 两个变体。实验结果显示，delimiter-only 相比 SOTA strict EM 下降 3.0 points，math-head-only 下降 1.7 points。值得注意的是，math-head-only 的平均 NFE 更高、block 更细，但性能仍低于 hybrid，说明性能提升并非仅来自更多计算或更细粒度划分，而是来自更正确的 boundary signal 以及 task-semantic 与 natural-language boundary 的互补融合。

## English Version

Full paragraph:

> To isolate the effect of boundary quality, we conduct a GSM8K ablation under the same B=32 decoding setup as our SOTA configuration. The full scheduler combines a learned math boundary head with natural-language newline delimiter cues. We compare it against two variants: a delimiter-only variant that removes the learned math boundary head, and a math-head-only variant that removes delimiter confidence from the hybrid score. The delimiter-only variant drops by 3.0 points in strict EM, while the math-head-only variant drops by 1.7 points. Notably, the math-head-only variant uses more average NFEs and produces more, shorter blocks, yet still underperforms the hybrid scheduler. This indicates that the improvement is not merely due to more computation or finer segmentation, but comes from the complementary fusion of task-semantic and natural-language boundary cues.

Short version:

> Removing either boundary source degrades GSM8K performance: delimiter-only drops by 3.0 points and math-head-only drops by 1.7 points. Since the math-head-only variant uses more NFEs but still underperforms, the hybrid gain cannot be explained by additional computation alone. These results suggest that task-semantic and natural-language boundary cues are complementary.

## Recommended Claim Strength

Recommended:

```text
Task-semantic and natural-language boundary cues are complementary; their hybridization yields the strongest reasoning performance.
```

Use carefully:

```text
More natural semantic boundaries improve performance.
```

Avoid overclaiming:

```text
Natural-language boundaries alone are always better.
```

The actual GSM8K result shows that natural-language delimiter alone is not enough. The strongest evidence is for complementary fusion.

## Connection to Earlier Boundary Overlap Result

Earlier boundary-overlap statistics showed that our learned/task semantic boundaries and AdaBlock-style delimiter boundaries are not identical. The GSM8K ablation adds a performance-level explanation:

- delimiter-only boundaries lose task-specific math structure;
- math-head-only boundaries become too fine and lose useful natural delimiter alignment;
- hybrid boundaries combine both sources and achieve the best EM.

This gives a clean narrative:

1. Our boundary scheduler is not simply reproducing AdaBlock.
2. Different boundary sources capture different structure.
3. Correctly fusing task-semantic and natural-language boundaries improves downstream generation quality.

