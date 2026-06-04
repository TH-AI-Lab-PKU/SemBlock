# IFEval Boundary Naturalness Curve

Naturalness is measured as exact Jaccard against the `prior_punctuation` trace on the same IFEval subset. Performance is final IFEval generation accuracy.

| Strategy | Boundary source | Samples | Naturalness Jaccard | Prompt strict | Inst strict | Prompt loose | Inst loose |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | fixed block | 55 | 0.074468 | 0.600000 | 0.666667 | 0.618182 | 0.690476 |
| adablock | CE/AdaBlock | 55 | 0.065306 | 0.618182 | 0.690476 | 0.636364 | 0.702381 |
| prior_punctuation | sentence/punctuation prior | 55 | 1.000000 | 0.636364 | 0.714286 | 0.636364 | 0.714286 |
| semantic_head | GUM semantic head | 55 | 0.069694 | 0.581818 | 0.666667 | 0.600000 | 0.690476 |
