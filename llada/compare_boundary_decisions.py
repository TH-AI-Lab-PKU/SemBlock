from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def load_trace_rows(path: Path) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _extract_positions(rows: Iterable[Dict[str, object]]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "unknown")
        position = row.get("selected_boundary_position")
        if position is None:
            continue
        grouped.setdefault(sample_id, []).append(int(position))
    for sample_id in grouped:
        grouped[sample_id] = sorted(grouped[sample_id])
    return grouped


def align_boundary_positions(
    adablock_rows: Iterable[Dict[str, object]],
    semantic_rows: Iterable[Dict[str, object]],
) -> Dict[str, Dict[str, List[int]]]:
    ada = _extract_positions(adablock_rows)
    semantic = _extract_positions(semantic_rows)
    sample_ids = sorted(set(ada) | set(semantic))
    return {
        sample_id: {
            "adablock_positions": ada.get(sample_id, []),
            "semantic_positions": semantic.get(sample_id, []),
        }
        for sample_id in sample_ids
    }


def _mean_min_distance(source: Sequence[int], target: Sequence[int]) -> float:
    if not source or not target:
        return 0.0
    total = 0.0
    for position in source:
        total += min(abs(position - other) for other in target)
    return total / len(source)


def compute_overlap_metrics(adablock_positions: Sequence[int], semantic_positions: Sequence[int]) -> Dict[str, float]:
    ada_set = set(int(position) for position in adablock_positions)
    semantic_set = set(int(position) for position in semantic_positions)
    overlap = len(ada_set & semantic_set)
    precision = overlap / len(semantic_set) if semantic_set else 0.0
    recall = overlap / len(ada_set) if ada_set else 0.0
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return {
        "adablock_count": float(len(ada_set)),
        "semantic_count": float(len(semantic_set)),
        "exact_overlap": float(overlap),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_semantic_to_adablock_distance": _mean_min_distance(sorted(semantic_set), sorted(ada_set)),
        "mean_adablock_to_semantic_distance": _mean_min_distance(sorted(ada_set), sorted(semantic_set)),
    }


def summarize_trace_overlap(
    adablock_rows: Iterable[Dict[str, object]],
    semantic_rows: Iterable[Dict[str, object]],
) -> List[Dict[str, object]]:
    aligned = align_boundary_positions(adablock_rows, semantic_rows)
    summaries: List[Dict[str, object]] = []
    for sample_id, grouped in aligned.items():
        metrics = compute_overlap_metrics(
            adablock_positions=grouped["adablock_positions"],
            semantic_positions=grouped["semantic_positions"],
        )
        metrics.update({
            "sample_id": sample_id,
            "adablock_positions": grouped["adablock_positions"],
            "semantic_positions": grouped["semantic_positions"],
        })
        summaries.append(metrics)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare AdaBlock and semantic scheduler boundary traces.")
    parser.add_argument("--adablock-trace", type=Path, required=True)
    parser.add_argument("--semantic-trace", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = summarize_trace_overlap(
        adablock_rows=load_trace_rows(args.adablock_trace),
        semantic_rows=load_trace_rows(args.semantic_trace),
    )
    rendered = json.dumps(summaries, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "sample_id",
                "adablock_count",
                "semantic_count",
                "exact_overlap",
                "precision",
                "recall",
                "f1",
                "mean_semantic_to_adablock_distance",
                "mean_adablock_to_semantic_distance",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in summaries:
                writer.writerow({key: row[key] for key in fieldnames})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
