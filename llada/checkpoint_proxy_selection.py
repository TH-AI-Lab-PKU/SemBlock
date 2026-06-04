from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


_PREFERRED_METRIC_KEYS = (
    "pass@1,none",
    "pass@1",
    "pass_at_1,none",
    "pass_at_1",
)

_TASK_ALIASES = {
    "humaneval": {
        "humaneval",
        "humaneval-50",
        "humaneval_50",
        "human_eval",
        "llada_humaneval_subset",
        "llada_humaneval_screening",
        "llada_humaneval_monitor",
    },
    "mbpp": {"mbpp", "mbpp-20", "mbpp_20", "llada_mbpp_subset"},
}


def load_results_payload(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _task_aliases(task_name: str) -> set[str]:
    normalized = str(task_name).strip().lower()
    aliases = {normalized}
    aliases.update(_TASK_ALIASES.get(normalized, set()))
    return aliases


def extract_proxy_metric(payload: dict[str, Any], task_name: str) -> float | None:
    task_aliases = _task_aliases(task_name)
    results = payload.get("results", payload)
    if not isinstance(results, dict):
        return None

    task_payload = None
    for key, value in results.items():
        if str(key).strip().lower() in task_aliases:
            task_payload = value
            break

    if not isinstance(task_payload, dict):
        return None

    normalized_metrics = {
        str(key).strip().lower(): value
        for key, value in task_payload.items()
        if isinstance(value, (int, float))
    }

    for metric_key in _PREFERRED_METRIC_KEYS:
        value = normalized_metrics.get(metric_key)
        if value is not None:
            return float(value)

    for key, value in normalized_metrics.items():
        if "pass@1" in key or "pass_at_1" in key:
            return float(value)
    return None


def extract_pass_at_1(payload: dict[str, Any], task_name: str) -> float | None:
    return extract_proxy_metric(payload, task_name)


def parse_optimizer_step_from_checkpoint_path(checkpoint_path: str | Path) -> int | None:
    match = re.search(r"boundary_head_step_(\d+)\.pt$", str(checkpoint_path))
    if match is None:
        return None
    return int(match.group(1))


def _iter_record_payloads(record: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for key in ("proxy_eval_results", "eval_results", "lm_eval_results", "results"):
        value = record.get(key)
        if isinstance(value, dict):
            yield value
    yield record


def extract_checkpoint_proxy_scores(record: dict[str, Any]) -> tuple[float | None, float | None]:
    humaneval = record.get("screening_humaneval_score", record.get("humaneval_score"))
    mbpp = record.get("mbpp_score")

    if isinstance(humaneval, (int, float)) and isinstance(mbpp, (int, float)):
        return float(humaneval), float(mbpp)

    extracted_humaneval = float(humaneval) if isinstance(humaneval, (int, float)) else None
    extracted_mbpp = float(mbpp) if isinstance(mbpp, (int, float)) else None
    for payload in _iter_record_payloads(record):
        if extracted_humaneval is None:
            extracted_humaneval = extract_proxy_metric(payload, "humaneval")
        if extracted_mbpp is None:
            extracted_mbpp = extract_proxy_metric(payload, "mbpp")
        if extracted_humaneval is not None and extracted_mbpp is not None:
            break
    return extracted_humaneval, extracted_mbpp


def _coerce_metric(record: dict[str, Any], metric_key: str, task_name: str) -> float:
    direct_value = record.get(metric_key)
    if isinstance(direct_value, (int, float)):
        return float(direct_value)

    for payload in _iter_record_payloads(record):
        extracted = extract_proxy_metric(payload, task_name)
        if extracted is not None:
            return float(extracted)
    return -1.0


def _coerce_optional_float(record: dict[str, Any], *metric_keys: str) -> float:
    for metric_key in metric_keys:
        value = record.get(metric_key)
        if isinstance(value, (int, float)):
            return float(value)
    return -1.0


def _coerce_positive_float(record: dict[str, Any], *metric_keys: str) -> float:
    value = _coerce_optional_float(record, *metric_keys)
    return value if value >= 0 else float(10**12)


def filter_checkpoint_records(
    records: Iterable[dict[str, Any]],
    *,
    min_boundary_precision: float = 0.80,
    max_boundary_positive_rate_ratio: float = 1.5,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for record in records:
        copied = dict(record)
        boundary_precision = _coerce_optional_float(copied, "valid_boundary_precision", "boundary_precision")
        positive_rate_ratio = _coerce_optional_float(
            copied,
            "boundary_positive_rate_ratio",
            "valid_boundary_positive_rate_ratio",
        )
        if boundary_precision >= 0 and boundary_precision < float(min_boundary_precision):
            continue
        if positive_rate_ratio >= 0 and positive_rate_ratio > float(max_boundary_positive_rate_ratio):
            continue
        filtered.append(copied)
    return filtered


def rank_checkpoint_records(
    records: Iterable[dict[str, Any]],
    *,
    preserve_earlier_optimizer_step: bool = True,
) -> list[dict[str, Any]]:
    candidate_records = filter_checkpoint_records(records)
    if not candidate_records:
        candidate_records = [dict(record) for record in records]

    def sort_key(record: dict[str, Any]) -> tuple[float, ...]:
        humaneval = _coerce_metric(record, "screening_humaneval_score", "humaneval")
        if humaneval < 0:
            humaneval = _coerce_metric(record, "humaneval_score", "humaneval")
        mbpp = _coerce_metric(record, "mbpp_score", "mbpp")
        parse_rate = _coerce_optional_float(
            record,
            "parse_rate",
            "valid_parse_rate",
            "completion_parse_rate",
            "humaneval_parse_rate",
            "proxy_parse_rate",
        )
        block_length_distribution_score = _coerce_optional_float(
            record,
            "block_length_distribution_score",
            "block_length_proxy_score",
        )
        block_length_distribution_error = _coerce_positive_float(
            record,
            "block_length_distribution_error",
            "block_length_emd",
            "block_length_kl",
            "avg_block_length_error",
            "mean_block_length_error",
        )
        avg_nfe = _coerce_positive_float(
            record,
            "avg_nfe",
            "mean_nfe",
            "average_nfe",
            "nfe_per_sample",
        )
        valid_boundary_precision = _coerce_optional_float(record, "valid_boundary_precision", "boundary_precision")
        valid_transition_f1 = _coerce_optional_float(record, "valid_transition_f1", "transition_f1")
        valid_phase_macro_f1 = _coerce_optional_float(record, "valid_phase_macro_f1", "phase_macro_f1")
        valid_boundary_f1 = _coerce_optional_float(record, "valid_boundary_f1", "boundary_f1")
        boundary_threshold = _coerce_optional_float(record, "boundary_threshold")
        boundary_window_ratio = _coerce_optional_float(record, "boundary_window_ratio")
        if boundary_window_ratio < 0:
            boundary_window_ratio = float(10**12)
        optimizer_step = record.get("optimizer_step")
        if optimizer_step is None:
            optimizer_step = parse_optimizer_step_from_checkpoint_path(record.get("checkpoint_path", ""))
        if preserve_earlier_optimizer_step:
            step_rank = int(optimizer_step) if optimizer_step is not None else 10**12
        else:
            step_rank = 0
        return (
            -humaneval,
            -mbpp,
            -parse_rate,
            -block_length_distribution_score,
            block_length_distribution_error,
            avg_nfe,
            -valid_boundary_precision,
            -valid_transition_f1,
            -valid_phase_macro_f1,
            -valid_boundary_f1,
            -boundary_threshold,
            boundary_window_ratio,
            step_rank,
        )

    return sorted(candidate_records, key=sort_key)


def build_proxy_ranking_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    ranked = rank_checkpoint_records(records)
    best_record = ranked[0] if ranked else None
    has_screening_metric = any(
        isinstance(record.get("screening_humaneval_score"), (int, float))
        for record in ranked
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_policy": "screening_proxy_eval" if has_screening_metric else "proxy_eval",
        "primary_metric": "screening_humaneval_score" if has_screening_metric else "humaneval_score",
        "secondary_metric": "mbpp_score",
        "tiebreaker_metrics": [
            "parse_rate",
            "block_length_distribution_score",
            "block_length_distribution_error",
            "avg_nfe",
            "valid_boundary_precision",
            "valid_transition_f1",
            "valid_phase_macro_f1",
            "valid_boundary_f1",
            "boundary_threshold",
            "boundary_window_ratio",
        ],
        "monitoring_metric": "monitor_humaneval_score" if has_screening_metric else None,
        "best_checkpoint_path": None if best_record is None else best_record.get("checkpoint_path"),
        "best_record": best_record,
        "records": ranked,
    }


def materialize_best_checkpoint(summary: dict[str, Any], output_path: str | Path) -> Path | None:
    best_checkpoint_path = summary.get("best_checkpoint_path")
    if not best_checkpoint_path and isinstance(summary.get("best_record"), dict):
        best_checkpoint_path = summary["best_record"].get("checkpoint_path")
    if not best_checkpoint_path:
        return None

    source_path = Path(best_checkpoint_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Best checkpoint does not exist: {source_path}")

    destination_path = Path(output_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if source_path.resolve() != destination_path.resolve():
        shutil.copy2(source_path, destination_path)
    return destination_path


def write_proxy_ranking_summary(path: str | Path, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    summary = build_proxy_ranking_summary(records)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary
