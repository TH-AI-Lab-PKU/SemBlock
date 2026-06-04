# GSM8K SOTA Boundary Ablation

This experiment tests the paper claim on GSM8K using the existing SOTA B=32 setup as the reference.

Reference:
- SOTA hybrid: math boundary head + natural-language newline delimiter.
- Existing limit-300 score: 85.33 strict EM.

Ablations:
- `delimiter_only_adablock`: removes the learned math boundary head and keeps the newline delimiter.
- `math_head_only`: removes delimiter confidence from the hybrid score by setting `boundary_prior_weight=1.0`.

The launcher defaults to GPUs `0,1` and avoids GPUs `2,3,4,5`.
