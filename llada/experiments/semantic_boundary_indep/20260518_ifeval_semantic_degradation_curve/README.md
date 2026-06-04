# IFEval Semantic Boundary Degradation Curve

This experiment directly tests the claim that generation quality improves when
boundaries stay closer to the learned natural-language semantic boundaries.

Setup:

- Task: IFEval local, first 55 examples.
- Clean baseline: existing GUM semantic-head trace and result from
  `20260516_boundary_overlap_tenth_corresponding`.
- Degradation variants: same GUM semantic head, same decoding config, but the
  selected block boundary is perturbed at runtime.
- Boundary fidelity: exact Jaccard against the clean semantic-head trace.
- Performance: final IFEval prompt/instance strict and loose accuracy.
- GPU policy: only GPUs 6 and 7 by default.

Variants:

- `clean`: no perturbation.
- `jitter2`: alternate moving boundaries by +/-2 tokens.
- `jitter4`: alternate moving boundaries by +/-4 tokens.
- `jitter8`: alternate moving boundaries by +/-8 tokens.
- `random`: replace each semantic boundary length with a random length within
  the block cap.
