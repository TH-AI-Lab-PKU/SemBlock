# GSM8K Boundary Ablation Paper Package

这个文件夹整理 GSM8K SOTA boundary ablation，供论文写作和结果追溯使用。

## 实验目的

证明论文中的核心观点：更正确、更贴近任务语义与自然语言语义边界的 block 划分，会带来更好的生成效果。  
在 GSM8K 上，我们用已经达到 SOTA 的 B=32 hybrid 配置作为 reference，然后分别移除 hybrid boundary scheduler 中的两个组成部分：

- `delimiter_only_adablock`: 只保留自然语言 newline delimiter，移除 learned math boundary head。
- `math_head_only`: 只保留 learned math boundary head，将 `boundary_prior_weight=1.0`，移除 delimiter confidence 对 hybrid score 的影响。
- `sota_hybrid`: learned math boundary head + natural-language newline delimiter，作为已有 SOTA reference。

## 主要结论

| Variant | Boundary signal | Limit | Strict EM | Delta vs SOTA | Avg NFE | Avg blocks | Avg block len |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sota_hybrid` | math head + NL delimiter | 300 | **0.8533** | 0.0000 | 104.48 | 26.87 | 19.06 |
| `delimiter_only_adablock` | NL delimiter only | 300 | 0.8233 | -0.0300 | 102.75 | 24.94 | 20.53 |
| `math_head_only` | math head only | 300 | 0.8367 | -0.0167 | 112.08 | 34.93 | 14.66 |

这个结果支持一个更稳妥的论文表述：

> On GSM8K, removing either component of the SOTA boundary scheduler degrades exact-match accuracy. The delimiter-only variant drops by 3.0 points, while the math-head-only variant drops by 1.7 points. This suggests that learned task-semantic boundaries and natural-language delimiter cues are complementary, and their hybridization yields the strongest reasoning performance.

## 文件说明

- `experiment_protocol.md`: 具体实验过程、配置、变量控制和结果来源。
- `results.md`: 论文视角结果分析，包括为什么这个 ablation 能支持观点。
- `paper_writing_notes.md`: 可直接改写进论文的英文/中文表述。
- `table_gsm8k_boundary_ablation.tex`: LaTeX 表格。
- `results_compact.csv`: 结果表 CSV。
- `reproduce_commands.sh`: 复现实验时可用的命令模板。

## 原始实验目录

原始脚本、日志、trace 和结果位于：

`llada/experiments/semantic_boundary_indep/20260519_gsm8k_sota_boundary_ablation/`

