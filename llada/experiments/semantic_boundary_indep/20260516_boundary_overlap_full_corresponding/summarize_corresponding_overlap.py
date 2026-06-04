#!/usr/bin/env python3
"""Summarize full corresponding boundary-overlap comparisons."""

from __future__ import annotations

import csv
import json
from pathlib import Path


TASKS = [
    ("HumanEval", "Code head", "humaneval_code_vs_adablock_all.json"),
    ("IFEval", "GUM semantic head", "ifeval_gum_vs_adablock_all.json"),
    ("GSM8K", "Math head", "gsm8k_math_vs_adablock_all.json"),
    ("MATH", "Math head", "math_math_vs_adablock_all.json"),
]


def _get(summary: dict, *keys: str) -> object:
    cur: object = summary
    for key in keys:
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(key, "")
    return cur


def main() -> None:
    root = Path(__file__).resolve().parent
    overlap_dir = root / "overlap"
    rows = []
    for task, head, filename in TASKS:
        path = overlap_dir / filename
        if not path.exists():
            rows.append(
                {
                    "task": task,
                    "semantic_head": head,
                    "status": "pending",
                    "matched_samples": "",
                    "semantic_boundaries": "",
                    "adablock_boundaries": "",
                    "exact_overlap_rate": "",
                    "jaccard": "",
                }
            )
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "task": task,
                "semantic_head": head,
                "status": "done",
                "matched_samples": _get(summary, "matched_samples"),
                "semantic_boundaries": _get(summary, "totals", "left_boundaries"),
                "adablock_boundaries": _get(summary, "totals", "right_boundaries"),
                "exact_overlap_rate": _get(summary, "rates", "left_overlap_rate_tolerance_0"),
                "jaccard": _get(summary, "rates", "jaccard_tolerance_0"),
            }
        )

    csv_path = root / "corresponding_overlap_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    md_path = root / "corresponding_overlap_summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| Task | Semantic head | Status | Matched samples | Semantic boundaries | AdaBlock boundaries | Exact overlap rate | Jaccard |\n")
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write(
                "| {task} | {semantic_head} | {status} | {matched_samples} | "
                "{semantic_boundaries} | {adablock_boundaries} | "
                "{exact_overlap_rate} | {jaccard} |\n".format(**row)
            )

    print(md_path)
    print(csv_path)


if __name__ == "__main__":
    main()
