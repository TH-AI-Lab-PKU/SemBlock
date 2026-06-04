# Full Corresponding Boundary Overlap

This experiment measures full-dataset boundary overlap between our semantic schedulers and AdaBlock under the corresponding SOTA head/task pairing.

The pairing is not a Cartesian product:

| Task | Our corresponding head |
| --- | --- |
| HumanEval | Code head |
| IFEval | GUM semantic head |
| GSM8K | Math head |
| MATH | Math head |

The comparison uses generated-token boundary offsets and excludes the terminal generation boundary by default, so the statistic reflects internal segmentation agreement rather than sequence length agreement.

Run:

```bash
bash llada/experiments/semantic_boundary_indep/20260516_boundary_overlap_full_corresponding/run_full_corresponding_overlap.sh
```

Primary outputs:

- `traces/<task>/<method>/rank_0.jsonl`: full trace events.
- `overlap/*_all.json`: aggregate overlap summaries.
- `overlap/*_all.samples.jsonl`: per-sample overlap summaries.
- `corresponding_overlap_summary.md`: compact task-level summary.
