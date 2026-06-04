from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT.parents[0] / "20260516_boundary_overlap_tenth_corresponding"

ROWS = [
    {
        "name": "clean",
        "mode": "none",
        "strength": 0,
        "result_dir": SOURCE_ROOT / "results" / "ifeval" / "gum_head_tenth",
        "overlap": None,
    },
    {
        "name": "jitter2",
        "mode": "jitter",
        "strength": 2,
        "result_dir": ROOT / "results" / "jitter2",
        "overlap": ROOT / "overlap" / "jitter2_vs_clean_semantic.json",
    },
    {
        "name": "jitter4",
        "mode": "jitter",
        "strength": 4,
        "result_dir": ROOT / "results" / "jitter4",
        "overlap": ROOT / "overlap" / "jitter4_vs_clean_semantic.json",
    },
    {
        "name": "jitter8",
        "mode": "jitter",
        "strength": 8,
        "result_dir": ROOT / "results" / "jitter8",
        "overlap": ROOT / "overlap" / "jitter8_vs_clean_semantic.json",
    },
    {
        "name": "random",
        "mode": "random",
        "strength": 1,
        "result_dir": ROOT / "results" / "random",
        "overlap": ROOT / "overlap" / "random_vs_clean_semantic.json",
    },
]


def latest_result(result_dir: Path) -> Path | None:
    files = sorted(result_dir.rglob("results_*.json"))
    return files[-1] if files else None


def nested(payload: dict[str, Any], *keys: str) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def load_metrics(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = nested(payload, "results", "ifeval_local") or {}
    samples = nested(payload, "n-samples", "ifeval_local") or {}
    return {
        "result_file": str(path),
        "samples": samples.get("effective"),
        "prompt_strict": metrics.get("prompt_level_strict_acc,none"),
        "inst_strict": metrics.get("inst_level_strict_acc,none"),
        "prompt_loose": metrics.get("prompt_level_loose_acc,none"),
        "inst_loose": metrics.get("inst_level_loose_acc,none"),
    }


def load_overlap(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "boundary_fidelity_jaccard": 1.0,
            "boundary_fidelity_precision": 1.0,
            "boundary_fidelity_recall": 1.0,
            "matched_samples": None,
        }
    if not path.exists():
        return {
            "boundary_fidelity_jaccard": None,
            "boundary_fidelity_precision": None,
            "boundary_fidelity_recall": None,
            "matched_samples": None,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    summaries = payload.get("task_summaries") or []
    if not summaries:
        return {}
    summary = summaries[0]
    exact = summary.get("exact") or {}
    return {
        "boundary_fidelity_jaccard": exact.get("jaccard"),
        "boundary_fidelity_precision": exact.get("semantic_to_adablock_precision"),
        "boundary_fidelity_recall": exact.get("adablock_to_semantic_recall"),
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
    for spec in ROWS:
        row = {
            "variant": spec["name"],
            "mode": spec["mode"],
            "strength": spec["strength"],
        }
        row.update(load_metrics(latest_result(spec["result_dir"])))
        row.update(load_overlap(spec["overlap"]))
        rows.append(row)

    fields = [
        "variant",
        "mode",
        "strength",
        "samples",
        "matched_samples",
        "boundary_fidelity_jaccard",
        "boundary_fidelity_precision",
        "boundary_fidelity_recall",
        "prompt_strict",
        "inst_strict",
        "prompt_loose",
        "inst_loose",
        "result_file",
    ]
    with (ROOT / "ifeval_semantic_degradation_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})

    with (ROOT / "ifeval_semantic_degradation_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# IFEval Semantic Boundary Degradation Curve\n\n")
        handle.write(
            "Boundary fidelity is exact Jaccard against the clean GUM semantic-head trace. "
            "Performance is final IFEval generation accuracy on the same 55-example subset.\n\n"
        )
        handle.write(
            "| Variant | Mode | Strength | Samples | Boundary fidelity Jaccard | "
            "Prompt strict | Inst strict | Prompt loose | Inst loose |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write(
                "| {variant} | {mode} | {strength} | {samples} | {jaccard} | "
                "{prompt_strict} | {inst_strict} | {prompt_loose} | {inst_loose} |\n".format(
                    variant=row["variant"],
                    mode=row["mode"],
                    strength=row["strength"],
                    samples=fmt(row.get("samples")),
                    jaccard=fmt(row.get("boundary_fidelity_jaccard")),
                    prompt_strict=fmt(row.get("prompt_strict")),
                    inst_strict=fmt(row.get("inst_strict")),
                    prompt_loose=fmt(row.get("prompt_loose")),
                    inst_loose=fmt(row.get("inst_loose")),
                )
            )


if __name__ == "__main__":
    main()
