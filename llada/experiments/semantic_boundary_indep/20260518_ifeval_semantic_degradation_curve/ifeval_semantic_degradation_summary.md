# IFEval Semantic Boundary Degradation Curve

Boundary fidelity is exact Jaccard against the clean GUM semantic-head trace. Performance is final IFEval generation accuracy on the same 55-example subset.

| Variant | Mode | Strength | Samples | Boundary fidelity Jaccard | Prompt strict | Inst strict | Prompt loose | Inst loose |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | none | 0 | 55 | 1.000000 | 0.581818 | 0.666667 | 0.600000 | 0.690476 |
| jitter2 | jitter | 2 | 55 | 0.190196 | 0.600000 | 0.678571 | 0.618182 | 0.690476 |
| jitter4 | jitter | 4 | 55 | 0.184397 | 0.600000 | 0.678571 | 0.618182 | 0.702381 |
| jitter8 | jitter | 8 | 55 | 0.177950 | 0.563636 | 0.630952 | 0.600000 | 0.654762 |
| random | random | 1 | 55 | 0.021399 | 0.654545 | 0.714286 | 0.709091 | 0.773810 |
