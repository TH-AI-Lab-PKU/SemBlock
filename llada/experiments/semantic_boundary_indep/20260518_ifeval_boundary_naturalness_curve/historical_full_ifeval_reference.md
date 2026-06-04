# Historical Full IFEval Reference

These are existing full IFEval B=32 results used as background for the
small-resource boundary-naturalness curve.

| Strategy | Samples | Prompt strict | Inst strict | Prompt loose | Inst loose |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed_full_b32 | 541 | 0.556377 | 0.642686 | 0.598891 | 0.682254 |
| adablock_full_b32 | 541 | 0.554529 | 0.643885 | 0.585952 | 0.677458 |
| semantic_head_thr075_full_b32 | 541 | 0.560074 | 0.657074 | 0.591497 | 0.689448 |

Takeaway: on the full set, the semantic head is slightly better on prompt
strict and instance-level metrics. The margin is modest, so the new curve
experiment focuses on whether higher naturalness Jaccard aligns with better
final IFEval accuracy under the same small subset and decoding config.
