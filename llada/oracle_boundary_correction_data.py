"""Helpers for oracle boundary correction records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Dict, Union


DeltaKey = Union[int, str]


def _parse_delta_key(raw_key) -> int:
    if isinstance(raw_key, bool):
        raise ValueError(f"Invalid delta key (bool not allowed): {raw_key}")
    if isinstance(raw_key, int):
        return raw_key
    if isinstance(raw_key, str):
        stripped = raw_key.strip()
        if stripped and stripped.lstrip("-").isdigit():
            return int(stripped)
        raise ValueError(f"Invalid delta key string: {raw_key}")
    if isinstance(raw_key, float):
        raise ValueError(f"Invalid delta key (float not allowed): {raw_key}")
    raise ValueError(f"Invalid delta key type: {type(raw_key).__name__}")


def _parse_boundary_index(raw_index) -> int:
    if isinstance(raw_index, bool):
        raise ValueError(f"Invalid boundary_index (bool not allowed): {raw_index}")
    if isinstance(raw_index, int):
        normalized = raw_index
    elif isinstance(raw_index, str):
        stripped = raw_index.strip()
        if stripped and stripped.lstrip("-").isdigit():
            normalized = int(stripped)
        else:
            raise ValueError(f"Invalid boundary_index string: {raw_index}")
    elif isinstance(raw_index, float):
        raise ValueError(f"Invalid boundary_index (float not allowed): {raw_index}")
    else:
        raise ValueError(f"Invalid boundary_index type: {type(raw_index).__name__}")
    if normalized < 0:
        raise ValueError(f"Invalid boundary_index (negative): {normalized}")
    return normalized


def _parse_score_value(raw_value) -> float:
    if isinstance(raw_value, bool):
        raise ValueError(f"Invalid delta score (bool not allowed): {raw_value}")
    try:
        score = float(raw_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid delta score type: {raw_value}") from exc
    if not math.isfinite(score):
        raise ValueError(f"Invalid delta score (non-finite): {raw_value}")
    return score


def _normalize_delta_scores(delta_scores) -> Dict[int, float]:
    if not isinstance(delta_scores, Mapping):
        raise ValueError("delta_scores must be a mapping")
    normalized: Dict[int, float] = {}
    for key, value in delta_scores.items():
        normalized_key = _parse_delta_key(key)
        if normalized_key in normalized:
            raise ValueError(
                f"Duplicate delta score key after normalization: {key} -> {normalized_key}"
            )
        normalized[normalized_key] = _parse_score_value(value)
    return normalized


def choose_best_delta_label(delta_scores: Mapping[DeltaKey, float]) -> int:
    normalized_scores = _normalize_delta_scores(delta_scores)
    if not normalized_scores:
        raise ValueError("delta_scores is empty")
    best_delta, _ = min(
        normalized_scores.items(),
        key=lambda item: (-item[1], abs(item[0]), item[0]),
    )
    return int(best_delta)


def classify_keep_vs_adjust(best_delta) -> str:
    normalized_best_delta = _parse_delta_key(best_delta)
    return "keep" if normalized_best_delta == 0 else "adjust"


def build_boundary_correction_record(
    sample_id,
    boundary_index,
    best_delta,
    delta_scores,
) -> dict:
    normalized_scores = _normalize_delta_scores(delta_scores)
    normalized_best_delta = _parse_delta_key(best_delta)
    normalized_boundary_index = _parse_boundary_index(boundary_index)
    if normalized_best_delta not in normalized_scores:
        raise ValueError(
            f"best_delta {normalized_best_delta} not present in normalized delta_scores"
        )
    return {
        "sample_id": sample_id,
        "boundary_index": normalized_boundary_index,
        "best_delta": normalized_best_delta,
        "gate_label": classify_keep_vs_adjust(normalized_best_delta),
        "delta_scores": normalized_scores,
    }
