# Results and Interpretation

## Main Result

| Variant | Boundary signal | Limit | Strict EM | Flexible EM | Delta vs SOTA | Avg NFE | Avg blocks | Avg block len | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `sota_hybrid` | math head + NL delimiter | 300 | **0.8533** | **0.8533** | 0.0000 | 104.48 | 26.87 | 19.06 | Best result |
| `delimiter_only_adablock` | NL delimiter only | 300 | 0.8233 | 0.8300 | -0.0300 | 102.75 | 24.94 | 20.53 | Removing learned math boundary head hurts |
| `math_head_only` | math head only | 300 | 0.8367 | 0.8400 | -0.0167 | 112.08 | 34.93 | 14.66 | Removing delimiter confidence also hurts |

## What This Shows

1. The SOTA hybrid boundary scheduler is best.

   `sota_hybrid` reaches 85.33 strict EM. Both ablations are worse, even though they keep the same model, task, block length, generation length, steps, cache setting, and GSM8K landing control.

2. Natural-language delimiter alone is insufficient for GSM8K reasoning.

   `delimiter_only_adablock` drops from 85.33 to 82.33 strict EM, a 3.00-point decrease. This supports the claim that a generic natural-language delimiter boundary is not enough for math reasoning; task-semantic boundary information matters.

3. Learned math boundary alone is also insufficient.

   `math_head_only` drops from 85.33 to 83.67 strict EM, a 1.67-point decrease. This suggests the learned math head benefits from being fused with natural-language delimiter cues.

4. The gain is not explained by more computation or simply using smaller blocks.

   `math_head_only` has higher Avg NFE than SOTA hybrid, 112.08 vs. 104.48, and more blocks, 34.93 vs. 26.87. Despite this, it still performs worse. Therefore, the advantage of `sota_hybrid` is not just caused by more denoising calls or finer segmentation. The boundary signal itself matters.

5. The result supports a complementary-boundary claim.

   The strongest interpretation is not that any single boundary source is universally best. The evidence supports that task-specific semantic boundaries and natural-language delimiter boundaries capture complementary structure, and their hybridization gives the best GSM8K reasoning performance.

## Recommended Paper Claim

Strong but safe:

> The GSM8K ablation shows that learned task-semantic boundaries and natural-language delimiter cues are complementary. Removing the math boundary head reduces strict EM by 3.0 points, while removing delimiter confidence reduces strict EM by 1.7 points. The hybrid scheduler achieves the best accuracy, even though the math-head-only variant uses more average NFEs and smaller blocks, indicating that boundary quality rather than computation alone drives the improvement.

More concise:

> The hybrid scheduler outperforms both delimiter-only and math-head-only variants, indicating that task-semantic and natural-language boundary cues provide complementary benefits.

## Suggested Figure/Table Caption

> Ablation of boundary signals on GSM8K using the SOTA B=32 decoding setup. Removing either the learned math boundary head or the natural-language delimiter confidence degrades exact match, showing that both task-semantic and natural-language boundary cues are necessary for the strongest reasoning performance.

