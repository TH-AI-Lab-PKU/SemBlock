# IFEval Boundary Naturalness Curve

Goal: test whether final generation performance improves as block boundaries
move closer to natural-language semantic boundaries.

Small-resource setup:

- Task: IFEval local, first 55 examples.
- Fixed generation config: `B=32`, `steps=16`, `gen_length=512`, cache on.
- Boundary sources:
  - `fixed`: fixed 32-token blocks.
  - `adablock`: CE/confidence delimiter boundary from the existing 55-example trace.
  - `prior_punctuation`: sentence/punctuation heuristic boundary, used as the natural-language proxy.
  - `semantic_head`: GUM semantic head from the existing 55-example trace.
- Naturalness proxy: exact boundary Jaccard against `prior_punctuation` on the same sample IDs.
- Performance: final IFEval prompt/instance strict and loose accuracy from lm-eval.

The runner waits for GPUs `2,3,4,5,6,7` only, and requires both enough free
memory and low utilization before launching a job.
