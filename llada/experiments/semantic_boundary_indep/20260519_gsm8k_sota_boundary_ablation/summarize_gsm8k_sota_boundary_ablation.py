#!/usr/bin/env python3
import json
import glob
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CLEAN_RESULT = Path(
    "/home/nvme01/workspace/AdaBlock-dLLM-main/llada/eval_results_math_semantic/"
    "aqua_gsm8k_confirm_l300_20260502/gsm8k/"
    "semantic_hybrid_b32_cache_on_limit300_thr0p60_minb8_selmax_score_above_threshold_mix0p70_landtrue/"
    "__home__nvme03__workspace__models__GSAI-ML__LLaDA-8B-Instruct/"
    "results_2026-05-03T00-19-41.222408.json"
)
CLEAN_LOG = Path(
    "/home/nvme01/workspace/AdaBlock-dLLM-main/llada/logs/math_semantic/"
    "aqua_gsm8k_confirm_l300_20260502/gsm8k/"
    "semantic_hybrid_b32_cache_on_limit300_thr0p60_minb8_selmax_score_above_threshold_mix0p70_landtrue.log"
)


VARIANTS = [
    {
        "name": "sota_hybrid",
        "boundary": "math head + NL delimiter",
        "result": CLEAN_RESULT,
        "log": CLEAN_LOG,
        "note": "Existing SOTA limit-300 reference",
    },
    {
        "name": "delimiter_only_adablock",
        "boundary": "NL delimiter only",
        "result_glob": ROOT / "results/delimiter_only_adablock/**/results_*.json",
        "log": ROOT / "logs/delimiter_only_adablock.log",
        "note": "Remove learned math boundary head",
    },
    {
        "name": "math_head_only",
        "boundary": "math head only",
        "result_glob": ROOT / "results/math_head_only/**/results_*.json",
        "log": ROOT / "logs/math_head_only.log",
        "note": "Remove delimiter confidence from the hybrid score",
    },
]


def latest_result(pattern: Path) -> Path | None:
    matches = sorted(glob.glob(str(pattern), recursive=True))
    return Path(matches[-1]) if matches else None


def read_metric(path: Path | None):
    if not path or not path.exists():
        return None
    data = json.loads(path.read_text())
    results = data.get("results", {}).get("gsm8k", {})
    return {
        "strict": results.get("exact_match,strict-match"),
        "flex": results.get("exact_match,flexible-extract"),
        "limit": data.get("config", {}).get("limit"),
    }


def read_stats(path: Path | None):
    if not path or not path.exists():
        return {}
    text = path.read_text(errors="ignore")
    stats = {}
    for key, pattern in [
        ("avg_nfe", r"Average NFE per sample:\s*([0-9.]+)"),
        ("avg_blocks", r"Average number of blocks per sample:\s*([0-9.]+)"),
        ("avg_block_len", r"Average block length:\s*([0-9.]+)"),
    ]:
        match = re.search(pattern, text)
        if match:
            stats[key] = float(match.group(1))
    return stats


def fmt(value, digits=4):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def main():
    rows = []
    clean_strict = None
    for item in VARIANTS:
        result = item.get("result")
        if result is None and item.get("result_glob") is not None:
            result = latest_result(item["result_glob"])
        metric = read_metric(result)
        stats = read_stats(item.get("log"))
        strict = metric["strict"] if metric else None
        if item["name"] == "sota_hybrid":
            clean_strict = strict
        delta = strict - clean_strict if strict is not None and clean_strict is not None else None
        rows.append((item, result, metric, stats, delta))

    md = [
        "# GSM8K SOTA Boundary Ablation",
        "",
        "All runs use the GSM8K SOTA B=32 decoding setup unless noted: LLaDA-8B-Instruct, "
        "`semantic_hybrid`, `steps=16`, `gen_length=512`, cache on, math head threshold 0.60, "
        "newline delimiter threshold 0.3, and `gsm8k_landing_control=true`.",
        "",
        "| Variant | Boundary signal | Limit | Strict EM | Flexible EM | Delta vs SOTA | Avg NFE | Avg blocks | Avg block len | Note |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    csv = ["variant,boundary,limit,strict_em,flex_em,delta_vs_sota,avg_nfe,avg_blocks,avg_block_len,note,result_path"]
    for item, result, metric, stats, delta in rows:
        limit = metric["limit"] if metric else None
        strict = metric["strict"] if metric else None
        flex = metric["flex"] if metric else None
        md.append(
            "| {name} | {boundary} | {limit} | {strict} | {flex} | {delta} | {avg_nfe} | {avg_blocks} | {avg_block_len} | {note} |".format(
                name=item["name"],
                boundary=item["boundary"],
                limit="" if limit is None else int(float(limit)),
                strict=fmt(strict),
                flex=fmt(flex),
                delta=fmt(delta),
                avg_nfe=fmt(stats.get("avg_nfe"), 2),
                avg_blocks=fmt(stats.get("avg_blocks"), 2),
                avg_block_len=fmt(stats.get("avg_block_len"), 2),
                note=item["note"],
            )
        )
        csv.append(
            ",".join(
                [
                    item["name"],
                    json.dumps(item["boundary"]),
                    "" if limit is None else str(int(float(limit))),
                    fmt(strict),
                    fmt(flex),
                    fmt(delta),
                    fmt(stats.get("avg_nfe"), 6),
                    fmt(stats.get("avg_blocks"), 6),
                    fmt(stats.get("avg_block_len"), 6),
                    json.dumps(item["note"]),
                    json.dumps(str(result) if result else ""),
                ]
            )
        )

    (ROOT / "gsm8k_sota_boundary_ablation_summary.md").write_text("\n".join(md) + "\n")
    (ROOT / "gsm8k_sota_boundary_ablation_summary.csv").write_text("\n".join(csv) + "\n")


if __name__ == "__main__":
    main()
