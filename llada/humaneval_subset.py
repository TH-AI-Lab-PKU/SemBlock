from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_HUMANEVAL_TOTAL_SIZE = 164
DEFAULT_HUMANEVAL_BUCKETS: tuple[tuple[int, int], ...] = (
    (0, 49),
    (50, 99),
    (100, 163),
)


def _normalize_bucket_ranges(
    bucket_ranges: Sequence[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    normalized = [
        (int(start), int(end))
        for start, end in (bucket_ranges or DEFAULT_HUMANEVAL_BUCKETS)
    ]
    if not normalized:
        raise ValueError("bucket_ranges must not be empty")

    for start, end in normalized:
        if start < 0 or end < start:
            raise ValueError("bucket_ranges must contain valid inclusive [start, end] pairs")
    return normalized


def _allocate_bucket_sample_sizes(
    bucket_sizes: Sequence[int],
    sample_size: int,
) -> list[int]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    total_size = sum(int(size) for size in bucket_sizes)
    if total_size <= 0:
        raise ValueError("bucket_sizes must sum to a positive value")
    if sample_size > total_size:
        raise ValueError("sample_size cannot exceed total_size")

    raw_targets = [sample_size * (int(size) / total_size) for size in bucket_sizes]
    allocations = [int(math.floor(value)) for value in raw_targets]
    remaining = sample_size - sum(allocations)

    remainders = sorted(
        (
            (raw_targets[index] - allocations[index], index)
            for index in range(len(bucket_sizes))
        ),
        key=lambda item: (-item[0], item[1]),
    )
    for _, index in remainders[:remaining]:
        allocations[index] += 1

    for index, size in enumerate(bucket_sizes):
        if allocations[index] > int(size):
            raise ValueError("bucket allocation exceeded bucket size")
    return allocations


def select_humaneval_indices(
    *,
    strategy: str,
    sample_size: int,
    seed: int,
    total_size: int = DEFAULT_HUMANEVAL_TOTAL_SIZE,
    bucket_ranges: Sequence[tuple[int, int]] | None = None,
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

    rng = random.Random(int(seed))
    if strategy_name == "prefix":
        return list(range(size))

    if strategy_name == "uniform":
        return sorted(rng.sample(range(total), size))

    if strategy_name == "harder_stratified":
        # Over-weight harder buckets: first bucket gets 1 part, remaining buckets get 2 parts each.
        # For 3 default buckets with sample_size=50: [10, 20, 20]
        normalized_buckets = _normalize_bucket_ranges(bucket_ranges)
        bucket_count = len(normalized_buckets)
        bucket_weights = [1] + [2] * (bucket_count - 1)
        total_weight = sum(bucket_weights)
        raw_targets = [size * w / total_weight for w in bucket_weights]
        allocations = [int(math.floor(t)) for t in raw_targets]
        remaining = size - sum(allocations)
        remainders = sorted(
            ((raw_targets[i] - allocations[i], i) for i in range(len(bucket_weights))),
            key=lambda item: (-item[0], item[1]),
        )
        for _, idx in remainders[:remaining]:
            allocations[idx] += 1
        selected: list[int] = []
        for (start, end), bucket_take in zip(normalized_buckets, allocations):
            if bucket_take <= 0:
                continue
            if end >= total:
                raise ValueError("bucket_ranges cannot exceed total_size")
            avail = list(range(start, min(end, total - 1) + 1))
            take = min(bucket_take, len(avail))
            selected.extend(rng.sample(avail, take))
        return sorted(selected)

    if strategy_name != "stratified":
        raise ValueError(f"Unsupported HumanEval subset strategy: {strategy}")

    normalized_buckets = _normalize_bucket_ranges(bucket_ranges)
    bucket_sizes = []
    for start, end in normalized_buckets:
        if end >= total:
            raise ValueError("bucket_ranges cannot exceed total_size")
        bucket_sizes.append(end - start + 1)

    allocations = _allocate_bucket_sample_sizes(bucket_sizes, size)
    selected: list[int] = []
    for (start, end), bucket_take in zip(normalized_buckets, allocations):
        if bucket_take <= 0:
            continue
        selected.extend(rng.sample(range(start, end + 1), bucket_take))
    return sorted(selected)


def build_humaneval_subset_manifest(
    *,
    output_path: str | Path,
    subset_label: str,
    strategy: str,
    sample_size: int,
    seed: int,
    total_size: int = DEFAULT_HUMANEVAL_TOTAL_SIZE,
    bucket_ranges: Sequence[tuple[int, int]] | None = None,
) -> dict[str, object]:
    indices = select_humaneval_indices(
        strategy=strategy,
        sample_size=sample_size,
        seed=seed,
        total_size=total_size,
        bucket_ranges=bucket_ranges,
    )
    normalized_buckets = _normalize_bucket_ranges(bucket_ranges)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subset_label": str(subset_label),
        "strategy": str(strategy),
        "sample_size": int(sample_size),
        "seed": int(seed),
        "total_size": int(total_size),
        "indices": indices,
        "task_ids": [f"HumanEval/{index}" for index in indices],
        "bucket_ranges": [
            {"start": int(start), "end": int(end)}
            for start, end in normalized_buckets
        ],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def load_humaneval_subset_manifest(path: str | Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    indices = payload.get("indices")
    if not isinstance(indices, list) or not indices:
        raise ValueError(f"Invalid HumanEval subset manifest: {path}")
    payload["indices"] = [int(index) for index in indices]
    return payload


def summarize_subset_coverage(
    indices: Iterable[int],
    *,
    bucket_ranges: Sequence[tuple[int, int]] | None = None,
) -> list[dict[str, int]]:
    normalized_buckets = _normalize_bucket_ranges(bucket_ranges)
    selected = [int(index) for index in indices]
    summary: list[dict[str, int]] = []
    for start, end in normalized_buckets:
        count = sum(1 for index in selected if start <= index <= end)
        summary.append(
            {
                "start": int(start),
                "end": int(end),
                "selected_count": int(count),
                "bucket_size": int(end - start + 1),
            }
        )
    return summary
