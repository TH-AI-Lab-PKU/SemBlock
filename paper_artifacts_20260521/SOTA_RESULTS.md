# SOTA Results for Paper

Date: 2026-05-21

This file collects the current paper-facing SOTA results for the three
specialized boundary-head lines:

- NL line: GUM semantic head, evaluated on IFEval.
- Math line: math head / semantic hybrid, evaluated on GSM8K and MATH.
- Code line: code head, evaluated on HumanEval.

MBPP is intentionally omitted.

## Main Results

| Task | Method | Model | B0=16 | B0=32 | B0=64 |
|---|---|---|---:|---:|---:|
| GSM8K | Ours, Math Head / Semantic Hybrid | LLaDA-Instruct | 80.36 | **82.60** | 78.77 |
| IFEval | Ours, GUM Semantic Head | LLaDA-Instruct | 56.00 | **56.93** | 51.20 |
| HumanEval | Ours, Code Head / Generalized Router v3 | LLaDA-Instruct | - | **46.95** | - |
| HumanEval | Ours, Code Head | LLaDA-1.5 | **47.56** | **49.39** | **50.00** |
| MATH | Ours, Math Head | LLaDA-Instruct | - | **37.80** | - |

## SOTA Cells Against the Baseline Table

| Task | Model | B0 | Ours | Best baseline in comparison table | Delta |
|---|---|---:|---:|---:|---:|
| GSM8K | LLaDA-Instruct | 32 | **82.60** | 80.60 | +2.00 |
| IFEval | LLaDA-Instruct | 32 | **56.93** | 55.64 | +1.29 |
| HumanEval | LLaDA-Instruct | 32 | **46.95** | 46.30 | +0.65 |
| HumanEval | LLaDA-1.5 | 16 | **47.56** | 39.00 | +8.56 |
| HumanEval | LLaDA-1.5 | 32 | **49.39** | 39.00 | +10.39 |
| HumanEval | LLaDA-1.5 | 64 | **50.00** | 38.40 | +11.60 |
| MATH | LLaDA-Instruct | 32 | **37.80** | 37.30 | +0.50 |

## Raw Result Files Included Here

The raw summaries copied into this paper folder are under `results_raw/`:

| File | Description |
|---|---|
| `results_raw/humaneval_llada15_b16_summary.csv` | HumanEval, LLaDA-1.5, B0=16, pass@1=47.56 |
| `results_raw/humaneval_llada15_b32_summary.csv` | HumanEval, LLaDA-1.5, B0=32, pass@1=49.39 |
| `results_raw/humaneval_llada15_b64_summary.csv` | HumanEval, LLaDA-1.5, B0=64, pass@1=50.00 |
| `results_raw/gsm8k_sota_boundary_ablation_summary.csv` | GSM8K math-head / semantic-hybrid ablation summary |
| `results_raw/ifeval_boundary_curve_summary.csv` | IFEval GUM semantic-head / natural-boundary summary |
| `../llada/experiments/semantic_boundary_indep/SOTA_ARCHIVE_20260512_generalized_v3_4695/results/full_humaneval_router_generalized_v3_sharded/merged_summary.json` | HumanEval, LLaDA-Instruct, B0=32, pass@1=46.95 |

## Notes

- GSM8K B0=16 and B0=64 are recorded measurements but are not the SOTA cells
  against the provided LLaDA-Instruct comparison table.
- IFEval B0=16 and B0=64 are recorded measurements for the GUM semantic-head
  line; the comparison table only reports B0=32 baselines for IFEval.
- HumanEval B0=16, B0=32, and B0=64 are all SOTA for LLaDA-1.5 in the provided
  comparison table.
- HumanEval LLaDA-Instruct B0=32 is also a SOTA cell. It comes from the
  generalized-router-v3 archive dated 2026-05-12 and scores 77/164, pass@1
  46.95, against the 46.30 baseline.
- MATH B0=32 uses the offline post-processing result reported as 37.80.
- Detailed hyperparameters are collected in `SOTA_HYPERPARAMETERS.md`.
