from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from build_task_conditioned_phase_boundary_jsonl import select_equidistant_indices


DEFAULT_MBPP_TOTAL_SIZE = 500


def select_mbpp_indices(
    *,
    strategy: str,
    sample_size: int,
    seed: int,
    total_size: int = DEFAULT_MBPP_TOTAL_SIZE,
) -> list[int]:
    strategy_name = str(strategy).strip().lower()
    total = int(total_size)
    size = int(sample_size)
    if size <= 0:
        raise ValueError("sample_size must be positive")
    if total <= 0:
        raise ValueError("total_size must be positive")
    if size > total:
        raise ValueError("sample_size cannot exceed total_size")

    if strategy_name == "prefix":
        return list(range(size))
    if strategy_name != "equidistant":
        raise ValueError(f"Unsupported MBPP subset strategy: {strategy}")
    return select_equidistant_indices(total, size)


def build_mbpp_subset_manifest(
    *,
    output_path: str | Path,
    subset_label: str,
    strategy: str,
    sample_size: int,
    seed: int,
    total_size: int = DEFAULT_MBPP_TOTAL_SIZE,
) -> dict[str, object]:
    indices = select_mbpp_indices(
        strategy=strategy,
        sample_size=sample_size,
        seed=seed,
        total_size=total_size,
    )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subset_label": str(subset_label),
        "strategy": str(strategy),
        "sample_size": int(sample_size),
        "seed": int(seed),
        "total_size": int(total_size),
        "indices": indices,
        "task_ids": [f"MBPP/{index}" for index in indices],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def load_mbpp_subset_manifest(path: str | Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    indices = payload.get("indices")
    if not isinstance(indices, list) or not indices:
        raise ValueError(f"Invalid MBPP subset manifest: {path}")
    payload["indices"] = [int(index) for index in indices]
    return payload


def summarize_subset_coverage(indices: Iterable[int], *, total_size: int = DEFAULT_MBPP_TOTAL_SIZE) -> dict[str, int]:
    selected = [int(index) for index in indices]
    if not selected:
        return {"total_size": int(total_size), "selected_count": 0, "min_index": -1, "max_index": -1}
    return {
        "total_size": int(total_size),
        "selected_count": len(selected),
        "min_index": min(selected),
        "max_index": max(selected),
    }
