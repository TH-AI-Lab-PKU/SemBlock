from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = (
    ROOT.parents[0]
    / "20260516_boundary_overlap_tenth_corresponding"
)


STRATEGIES = [
    {
        "name": "fixed",
        "boundary_source": "fixed block",
        "result_dir": ROOT / "results" / "fixed",
        "overlap": ROOT / "overlap" / "fixed_vs_prior_punctuation.json",
    },
    {
        "name": "adablock",
        "boundary_source": "CE/AdaBlock",
        "result_dir": SOURCE_ROOT / "results" / "ifeval" / "adablock_tenth",
        "overlap": ROOT / "overlap" / "adablock_vs_prior_punctuation.json",
    },
    {
        "name": "prior_punctuation",
        "boundary_source": "sentence/punctuation prior",
        "result_dir": ROOT / "results" / "prior_punctuation",
        "overlap": None,
    },
    {
        "name": "semantic_head",
        "boundary_source": "GUM semantic head",
        "result_dir": SOURCE_ROOT / "results" / "ifeval" / "gum_head_tenth",
        "overlap": ROOT / "overlap" / "semantic_head_vs_prior_punctuation.json",
    },
]


def latest_result(result_dir: Path) -> Path | None:
    files = sorted(result_dir.rglob("results_*.json"))
    return files[-1] if files else None


def get_nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def load_metrics(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = get_nested(payload, "results", "ifeval_local") or {}
    samples = get_nested(payload, "n-samples", "ifeval_local") or {}
    return {
        "result_file": str(path),
        "effective_samples": samples.get("effective"),
        "prompt_strict": metrics.get("prompt_level_strict_acc,none"),
        "inst_strict": metrics.get("inst_level_strict_acc,none"),
        "prompt_loose": metrics.get("prompt_level_loose_acc,none"),
        "inst_loose": metrics.get("inst_level_loose_acc,none"),
    }


def load_overlap(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "naturalness_jaccard": 1.0,
            "naturalness_precision": 1.0,
            "naturalness_recall": 1.0,
            "matched_samples": None,
        }
    if not path.exists():
        return {
            "naturalness_jaccard": None,
            "naturalness_precision": None,
            "naturalness_recall": None,
            "matched_samples": None,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    summaries = payload.get("task_summaries") or []
    if not summaries:
        return {}
    summary = summaries[0]
    exact = summary.get("exact") or {}
    return {
        "naturalness_jaccard": exact.get("jaccard"),
        "naturalness_precision": exact.get("semantic_to_adablock_precision"),
        "naturalness_recall": exact.get("adablock_to_semantic_recall"),
        "matched_samples": summary.get("shared_sample_count"),
    }


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    rows: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        result_file = latest_result(strategy["result_dir"])
        row = {
            "strategy": strategy["name"],
            "boundary_source": strategy["boundary_source"],
        }
        row.update(load_metrics(result_file))
        row.update(load_overlap(strategy["overlap"]))
        rows.append(row)

    csv_path = ROOT / "ifeval_boundary_curve_summary.csv"
    fieldnames = [
        "strategy",
        "boundary_source",
        "effective_samples",
        "naturalness_jaccard",
        "naturalness_precision",
        "naturalness_recall",
        "matched_samples",
        "prompt_strict",
        "inst_strict",
        "prompt_loose",
        "inst_loose",
        "result_file",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    md_path = ROOT / "ifeval_boundary_curve_summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# IFEval Boundary Naturalness Curve\n\n")
        handle.write(
            "Naturalness is measured as exact Jaccard against the "
            "`prior_punctuation` trace on the same IFEval subset. "
            "Performance is final IFEval generation accuracy.\n\n"
        )
        handle.write(
            "| Strategy | Boundary source | Samples | Naturalness Jaccard | "
            "Prompt strict | Inst strict | Prompt loose | Inst loose |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write(
                "| {strategy} | {boundary_source} | {effective_samples} | "
                "{naturalness_jaccard} | {prompt_strict} | {inst_strict} | "
                "{prompt_loose} | {inst_loose} |\n".format(
                    strategy=row["strategy"],
                    boundary_source=row["boundary_source"],
                    effective_samples=fmt(row.get("effective_samples")),
                    naturalness_jaccard=fmt(row.get("naturalness_jaccard")),
                    prompt_strict=fmt(row.get("prompt_strict")),
                    inst_strict=fmt(row.get("inst_strict")),
                    prompt_loose=fmt(row.get("prompt_loose")),
                    inst_loose=fmt(row.get("inst_loose")),
                )
            )

    print(md_path)
    print(csv_path)


if __name__ == "__main__":
    main()
