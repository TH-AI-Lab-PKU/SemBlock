from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class TraceBoundary:
    offset: int
    accepted: bool
    block_length: int
    step_index: int
    scheduler_name: str


@dataclass(frozen=True)
class SampleBoundarySet:
    sample_key: str
    sample_id: str | None
    request_index: str | None
    boundaries: tuple[TraceBoundary, ...]

    @property
    def all_offsets(self) -> tuple[int, ...]:
        return tuple(boundary.offset for boundary in self.boundaries)

    @property
    def accepted_offsets(self) -> tuple[int, ...]:
        return tuple(boundary.offset for boundary in self.boundaries if boundary.accepted)


def _iter_trace_files(path: str | Path) -> Iterable[Path]:
    root = Path(path)
    if root.is_file():
        yield root
        return
    if not root.exists():
        raise FileNotFoundError(f"Trace path does not exist: {root}")
    yield from sorted(root.rglob("rank_*.jsonl"))


def _expand_trace_or_block_history_record(
    payload: Mapping[str, object],
    *,
    record_index: int,
) -> list[dict[str, object]]:
    if "selected_block_length" in payload:
        return [dict(payload)]

    block_history = payload.get("block_history")
    if not isinstance(block_history, list):
        return []

    sample_id = payload.get("sample_id") or payload.get("task_id") or payload.get("id") or f"sample-{record_index}"
    request_index = payload.get("request_index", record_index)
    scheduler_name = payload.get("scheduler_name") or payload.get("cache_policy") or payload.get("cache_mode") or "block_history"
    expanded: list[dict[str, object]] = []
    generated_length = 0
    for step_index, raw_block_length in enumerate(block_history):
        try:
            block_length = int(raw_block_length)
        except (TypeError, ValueError):
            continue
        if block_length <= 0:
            continue
        expanded.append(
            {
                "scheduler_name": str(scheduler_name),
                "sample_id": str(sample_id),
                "request_index": request_index,
                "step_index": step_index,
                "generated_length": generated_length,
                "selected_block_length": block_length,
                "selected_boundary_index": block_length - 1,
                "metadata": {
                    "source_record_had_block_history": True,
                    "source_task_name": payload.get("task_name"),
                },
            }
        )
        generated_length += block_length
    return expanded


def _load_jsonl_trace_records(path: str | Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    record_index = 0
    for trace_path in _iter_trace_files(path):
        with open(trace_path, "r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {trace_path}:{line_number}") from exc
                if isinstance(payload, dict):
                    expanded = _expand_trace_or_block_history_record(payload, record_index=record_index)
                    records.extend(expanded)
                    record_index += 1
    return records


def _stable_key(record: Mapping[str, object], match_key: str) -> str:
    if match_key == "sample_id":
        value = record.get("sample_id")
    elif match_key == "request_index":
        value = record.get("request_index")
    else:
        raise ValueError(f"Unsupported match_key: {match_key}")
    if value is None:
        raise ValueError(f"Trace record is missing {match_key}: {record}")
    return str(value)


def _choose_match_key(left_records: Sequence[Mapping[str, object]], right_records: Sequence[Mapping[str, object]]) -> str:
    left_sample_ids = {str(record.get("sample_id")) for record in left_records if record.get("sample_id") is not None}
    right_sample_ids = {str(record.get("sample_id")) for record in right_records if record.get("sample_id") is not None}
    sample_overlap = left_sample_ids & right_sample_ids
    if sample_overlap and len(sample_overlap) >= max(1, min(len(left_sample_ids), len(right_sample_ids)) // 2):
        return "sample_id"
    return "request_index"


def _records_to_boundary_sets(
    records: Sequence[Mapping[str, object]],
    *,
    match_key: str,
    include_terminal: bool,
    source: str,
) -> dict[str, SampleBoundarySet]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[_stable_key(record, match_key)].append(record)

    sample_sets: dict[str, SampleBoundarySet] = {}
    for sample_key, sample_records in grouped.items():
        ordered = sorted(
            sample_records,
            key=lambda record: (
                int(record.get("step_index", 0) or 0),
                int(record.get("generated_length", 0) or 0),
            ),
        )
        boundaries: list[TraceBoundary] = []
        for record in ordered:
            generated_length = int(record.get("generated_length", 0) or 0)
            block_length = int(record.get("selected_block_length", 0) or 0)
            if block_length <= 0:
                continue
            accepted = record.get("selected_boundary_index") is not None
            if source == "accepted" and not accepted:
                continue
            boundaries.append(
                TraceBoundary(
                    offset=generated_length + block_length,
                    accepted=accepted,
                    block_length=block_length,
                    step_index=int(record.get("step_index", 0) or 0),
                    scheduler_name=str(record.get("scheduler_name") or ""),
                )
            )

        if boundaries and not include_terminal:
            terminal_offset = max(boundary.offset for boundary in boundaries)
            boundaries = [boundary for boundary in boundaries if boundary.offset != terminal_offset]

        first = ordered[0] if ordered else {}
        sample_sets[sample_key] = SampleBoundarySet(
            sample_key=sample_key,
            sample_id=None if first.get("sample_id") is None else str(first.get("sample_id")),
            request_index=None if first.get("request_index") is None else str(first.get("request_index")),
            boundaries=tuple(boundaries),
        )
    return sample_sets


def _count_within_tolerance(query: Sequence[int], reference: Sequence[int], tolerance: int) -> int:
    if not query or not reference:
        return 0
    reference_sorted = sorted(set(int(value) for value in reference))
    matched = 0
    for value in sorted(set(int(value) for value in query)):
        nearest = min(abs(value - candidate) for candidate in reference_sorted)
        if nearest <= tolerance:
            matched += 1
    return matched


def _nearest_distances(query: Sequence[int], reference: Sequence[int]) -> list[int]:
    if not query or not reference:
        return []
    reference_values = sorted(set(int(value) for value in reference))
    return [
        min(abs(int(value) - candidate) for candidate in reference_values)
        for value in sorted(set(int(value) for value in query))
    ]


def _safe_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * float(percentile)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _summarize_one_pair(
    *,
    task_name: str,
    semantic_sets: Mapping[str, SampleBoundarySet],
    adablock_sets: Mapping[str, SampleBoundarySet],
    tolerances: Sequence[int],
) -> dict[str, object]:
    common_keys = sorted(set(semantic_sets) & set(adablock_sets))
    if not common_keys:
        raise ValueError(f"No shared samples for task={task_name}")

    per_sample: list[dict[str, object]] = []
    semantic_counts: list[int] = []
    adablock_counts: list[int] = []
    exact_intersections = 0
    exact_unions = 0
    total_semantic = 0
    total_adablock = 0
    nearest_semantic_to_adablock: list[int] = []
    nearest_adablock_to_semantic: list[int] = []
    tolerance_totals = {
        int(tolerance): {
            "semantic_matches": 0,
            "adablock_matches": 0,
        }
        for tolerance in tolerances
    }

    for sample_key in common_keys:
        semantic = semantic_sets[sample_key]
        adablock = adablock_sets[sample_key]
        semantic_offsets = set(semantic.all_offsets)
        adablock_offsets = set(adablock.all_offsets)
        intersection = semantic_offsets & adablock_offsets
        union = semantic_offsets | adablock_offsets
        total_semantic += len(semantic_offsets)
        total_adablock += len(adablock_offsets)
        exact_intersections += len(intersection)
        exact_unions += len(union)
        semantic_counts.append(len(semantic_offsets))
        adablock_counts.append(len(adablock_offsets))

        sem_to_ada = _nearest_distances(sorted(semantic_offsets), sorted(adablock_offsets))
        ada_to_sem = _nearest_distances(sorted(adablock_offsets), sorted(semantic_offsets))
        nearest_semantic_to_adablock.extend(sem_to_ada)
        nearest_adablock_to_semantic.extend(ada_to_sem)
        tolerance_rows: dict[str, object] = {}
        for tolerance in tolerances:
            tolerance = int(tolerance)
            semantic_matches = _count_within_tolerance(sorted(semantic_offsets), sorted(adablock_offsets), tolerance)
            adablock_matches = _count_within_tolerance(sorted(adablock_offsets), sorted(semantic_offsets), tolerance)
            tolerance_totals[tolerance]["semantic_matches"] += semantic_matches
            tolerance_totals[tolerance]["adablock_matches"] += adablock_matches
            tolerance_rows[str(tolerance)] = {
                "semantic_to_adablock_precision": semantic_matches / len(semantic_offsets) if semantic_offsets else None,
                "adablock_to_semantic_recall": adablock_matches / len(adablock_offsets) if adablock_offsets else None,
            }

        per_sample.append(
            {
                "sample_key": sample_key,
                "semantic_sample_id": semantic.sample_id,
                "adablock_sample_id": adablock.sample_id,
                "semantic_boundary_count": len(semantic_offsets),
                "adablock_boundary_count": len(adablock_offsets),
                "exact_intersection_count": len(intersection),
                "exact_jaccard": len(intersection) / len(union) if union else None,
                "semantic_offsets": sorted(semantic_offsets),
                "adablock_offsets": sorted(adablock_offsets),
                "nearest_semantic_to_adablock_mean": _safe_mean(sem_to_ada),
                "nearest_adablock_to_semantic_mean": _safe_mean(ada_to_sem),
                "tolerance": tolerance_rows,
            }
        )

    tolerance_summary = {}
    for tolerance, totals in tolerance_totals.items():
        tolerance_summary[str(tolerance)] = {
            "semantic_to_adablock_precision": (
                totals["semantic_matches"] / total_semantic if total_semantic else None
            ),
            "adablock_to_semantic_recall": (
                totals["adablock_matches"] / total_adablock if total_adablock else None
            ),
        }

    return {
        "task_name": task_name,
        "shared_sample_count": len(common_keys),
        "semantic_boundary_count": total_semantic,
        "adablock_boundary_count": total_adablock,
        "semantic_boundaries_per_sample_mean": _safe_mean(semantic_counts),
        "adablock_boundaries_per_sample_mean": _safe_mean(adablock_counts),
        "semantic_to_adablock_count_ratio": (
            total_semantic / total_adablock if total_adablock else None
        ),
        "exact": {
            "intersection_count": exact_intersections,
            "union_count": exact_unions,
            "jaccard": exact_intersections / exact_unions if exact_unions else None,
            "semantic_to_adablock_precision": exact_intersections / total_semantic if total_semantic else None,
            "adablock_to_semantic_recall": exact_intersections / total_adablock if total_adablock else None,
        },
        "tolerance": tolerance_summary,
        "nearest_distance": {
            "semantic_to_adablock_mean": _safe_mean(nearest_semantic_to_adablock),
            "semantic_to_adablock_median": _safe_median(nearest_semantic_to_adablock),
            "semantic_to_adablock_p90": _percentile(nearest_semantic_to_adablock, 0.90),
            "adablock_to_semantic_mean": _safe_mean(nearest_adablock_to_semantic),
            "adablock_to_semantic_median": _safe_median(nearest_adablock_to_semantic),
            "adablock_to_semantic_p90": _percentile(nearest_adablock_to_semantic, 0.90),
        },
        "per_sample": per_sample,
    }


def _compact_task_summary(task_summary: Mapping[str, object]) -> dict[str, object]:
    exact = task_summary.get("exact") or {}
    nearest = task_summary.get("nearest_distance") or {}
    return {
        "task_name": task_summary.get("task_name"),
        "shared_sample_count": task_summary.get("shared_sample_count"),
        "semantic_boundary_count": task_summary.get("semantic_boundary_count"),
        "adablock_boundary_count": task_summary.get("adablock_boundary_count"),
        "semantic_to_adablock_count_ratio": task_summary.get("semantic_to_adablock_count_ratio"),
        "exact_jaccard": exact.get("jaccard") if isinstance(exact, Mapping) else None,
        "exact_semantic_to_adablock_precision": (
            exact.get("semantic_to_adablock_precision") if isinstance(exact, Mapping) else None
        ),
        "exact_adablock_to_semantic_recall": (
            exact.get("adablock_to_semantic_recall") if isinstance(exact, Mapping) else None
        ),
        "nearest_semantic_to_adablock_mean": (
            nearest.get("semantic_to_adablock_mean") if isinstance(nearest, Mapping) else None
        ),
        "nearest_adablock_to_semantic_mean": (
            nearest.get("adablock_to_semantic_mean") if isinstance(nearest, Mapping) else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare learned semantic scheduler boundaries against AdaBlock boundaries "
            "from scheduler trace JSONL files."
        )
    )
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("TASK", "SEMANTIC_TRACE", "ADABLOCK_TRACE"),
        required=True,
        help=(
            "Task name plus trace file/dir for semantic and AdaBlock. May be repeated, "
            "e.g. --pair humaneval sem/traces ada/traces."
        ),
    )
    parser.add_argument(
        "--match-key",
        choices=["auto", "sample_id", "request_index"],
        default="auto",
        help="How to align samples across the two trace sets. Auto prefers sample_id when it overlaps.",
    )
    parser.add_argument(
        "--semantic-source",
        choices=["all", "accepted"],
        default="all",
        help="Use every semantic block cut, or only semantic cuts whose selected_boundary_index is not null.",
    )
    parser.add_argument(
        "--adablock-source",
        choices=["all", "accepted"],
        default="all",
        help="Use every AdaBlock cut, or only trace cuts whose selected_boundary_index is not null.",
    )
    parser.add_argument(
        "--tolerances",
        type=str,
        default="0,1,2,4,8",
        help="Comma-separated token tolerances for near-overlap precision/recall.",
    )
    parser.add_argument(
        "--include-terminal",
        action="store_true",
        help="Include each sample's final generated-token boundary. Default excludes it to avoid inflated overlap.",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON summary output path.")
    parser.add_argument(
        "--per-sample-output",
        type=Path,
        default=None,
        help="Optional JSONL path for per-sample comparison rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tolerances = [int(part.strip()) for part in str(args.tolerances).split(",") if part.strip()]
    task_summaries: list[dict[str, object]] = []
    per_sample_rows: list[dict[str, object]] = []

    for task_name, semantic_trace_path, adablock_trace_path in args.pair:
        semantic_records = _load_jsonl_trace_records(semantic_trace_path)
        adablock_records = _load_jsonl_trace_records(adablock_trace_path)
        if not semantic_records:
            raise ValueError(f"No semantic trace records found under {semantic_trace_path}")
        if not adablock_records:
            raise ValueError(f"No AdaBlock trace records found under {adablock_trace_path}")

        match_key = args.match_key
        if match_key == "auto":
            match_key = _choose_match_key(semantic_records, adablock_records)

        semantic_sets = _records_to_boundary_sets(
            semantic_records,
            match_key=match_key,
            include_terminal=bool(args.include_terminal),
            source=args.semantic_source,
        )
        adablock_sets = _records_to_boundary_sets(
            adablock_records,
            match_key=match_key,
            include_terminal=bool(args.include_terminal),
            source=args.adablock_source,
        )
        summary = _summarize_one_pair(
            task_name=task_name,
            semantic_sets=semantic_sets,
            adablock_sets=adablock_sets,
            tolerances=tolerances,
        )
        summary["semantic_trace_path"] = str(semantic_trace_path)
        summary["adablock_trace_path"] = str(adablock_trace_path)
        summary["match_key"] = str(match_key)
        summary["semantic_source"] = str(args.semantic_source)
        summary["adablock_source"] = str(args.adablock_source)
        summary["include_terminal"] = bool(args.include_terminal)
        for row in summary.get("per_sample", []):
            if isinstance(row, dict):
                per_sample_rows.append({"task_name": task_name, **row})
        task_summaries.append(summary)

    compact = [_compact_task_summary(task_summary) for task_summary in task_summaries]
    output_payload = {
        "summary_type": "semantic_vs_adablock_boundary_overlap",
        "task_summaries": task_summaries,
        "compact_task_summaries": compact,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.per_sample_output is not None:
        args.per_sample_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.per_sample_output, "w", encoding="utf-8") as handle:
            for row in per_sample_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
