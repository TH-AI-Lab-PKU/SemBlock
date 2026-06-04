# Semantic Boundary vs AdaBlock Boundary Overlap

This folder contains the first overlap measurements for the boundary-ablation
thread.  The comparison is done in generated-token offset space, not absolute
sequence position, so prompt-length differences do not inflate or deflate the
match rate.  The terminal boundary of each sample is excluded by default.

## HumanEval

Compared traces:

- Semantic: `SOTA_ARCHIVE_20260512_generalized_v3_4695/results/full_humaneval_router_generalized_v3_sharded`
- AdaBlock: `20260510_adablock_he32_same_subset_dual_cache/traces`
- Shared samples: 32

All scheduler cuts:

| Metric | Value |
|---|---:|
| Semantic boundary count | 714 |
| AdaBlock boundary count | 1015 |
| Count ratio | 0.7034 |
| Exact Jaccard | 0.0341 |
| Semantic-to-AdaBlock exact precision | 0.0798 |
| AdaBlock-to-semantic exact recall | 0.0562 |
| Mean nearest distance, semantic to AdaBlock | 6.62 tokens |
| Mean nearest distance, AdaBlock to semantic | 6.81 tokens |

Accepted semantic cuts only:

| Metric | Value |
|---|---:|
| Semantic boundary count | 347 |
| AdaBlock boundary count | 1015 |
| Count ratio | 0.3419 |
| Exact Jaccard | 0.0172 |
| Semantic-to-AdaBlock exact precision | 0.0663 |
| AdaBlock-to-semantic exact recall | 0.0227 |
| Mean nearest distance, semantic to AdaBlock | 6.78 tokens |
| Mean nearest distance, AdaBlock to semantic | 44.26 tokens |

The accepted-cut view is the cleaner evidence for the claim that the learned
semantic boundary is not merely AdaBlock delimiter confidence.  Its exact
Jaccard is only 1.72%, and even semantic-to-AdaBlock exact precision is 6.63%.

## Files

- `humaneval_sota_vs_adablock_all.json`
- `humaneval_sota_vs_adablock_all.samples.jsonl`
- `humaneval_sota_vs_adablock_accepted.json`
- `humaneval_sota_vs_adablock_accepted.samples.jsonl`
- `humaneval_proxy_vs_adablock_all.json`

## Current Gaps

No paired AdaBlock trace was found locally for GSM8K, MATH, or IFEval in this
worktree.  The math oracle runs contain `block_history`, but those are oracle
or natural-language semantic blocks rather than AdaBlock delimiter boundaries,
so they should not be used as the AdaBlock side of this comparison.

Once paired traces are produced for those tasks, the same comparison command can
be reused:

```bash
python llada/compare_boundary_traces.py \
  --pair TASK_NAME SEMANTIC_TRACE_DIR ADABLOCK_TRACE_DIR \
  --match-key sample_id \
  --output llada/experiments/semantic_boundary_indep/20260516_boundary_overlap/TASK_NAME_all.json \
  --per-sample-output llada/experiments/semantic_boundary_indep/20260516_boundary_overlap/TASK_NAME_all.samples.jsonl
```
