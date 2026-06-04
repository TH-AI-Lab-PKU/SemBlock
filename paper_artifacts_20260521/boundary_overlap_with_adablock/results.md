# Results and Interpretation

## Result Table

| Task | Semantic head | Matched samples | Exact overlap | Jaccard | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| HumanEval | Code head | 17 | 4.76% | 1.95% | Very different from AdaBlock |
| IFEval | GUM semantic head | 55 | 25.88% | 11.25% | Low overlap with AdaBlock |
| GSM8K | Math head | 132 | 65.45% | 51.35% | Partially aligned, but still non-identical |
| MATH | Math head | 351 | 82.39% | 69.19% | More aligned; partial result |

## Interpretation

The overlap is very low on HumanEval and IFEval. This is the cleanest evidence that the learned semantic heads are not simply selecting the same boundaries as AdaBlock.

For GSM8K and MATH, the overlap is higher. This is expected because math solutions often contain explicit step boundaries, line breaks, equations, and final-answer structures. Those structures can be detected by both semantic heads and delimiter-based AdaBlock. However, the boundary sets are still not identical: GSM8K has only 51.35% Jaccard overlap, and even MATH remains below perfect agreement.

The strongest paper claim is:

> Our semantic scheduler does not merely reproduce AdaBlock boundaries. It differs strongly on code and instruction-following tasks, and remains non-identical even on math tasks where natural delimiter cues partially align with reasoning steps.

## How This Connects to the GSM8K Ablation

This overlap table establishes that the boundary sources are different. The GSM8K boundary ablation then shows that the difference matters for performance:

- delimiter-only / AdaBlock-like boundary underperforms SOTA hybrid by 3.0 strict EM points;
- math-head-only boundary underperforms SOTA hybrid by 1.67 strict EM points;
- hybrid performs best, showing that task-semantic and natural-language boundary cues are complementary.

Together, the two experiments form a clean paper narrative:

1. Boundary schedules are not identical to AdaBlock.
2. Task-semantic and natural-language boundary cues capture different structure.
3. Combining them gives better downstream performance.

## Recommended Use in Paper

Use this table before or alongside the GSM8K ablation. The overlap table supports the novelty/difference claim; the GSM8K ablation supports the performance claim.

Suggested transition:

> We first quantify whether our semantic scheduler simply recovers AdaBlock boundaries. The overlap is low on HumanEval and IFEval and remains imperfect on math tasks. We then ablate the two boundary sources on GSM8K and show that their hybridization yields the best exact-match accuracy.

