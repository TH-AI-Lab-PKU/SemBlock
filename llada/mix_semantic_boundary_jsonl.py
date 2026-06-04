from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _parse_csv(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def _parse_weights(raw_value: str | None, count: int) -> list[float]:
    if not raw_value:
        return [1.0 / max(count, 1)] * count
    values = [float(part.strip()) for part in raw_value.split(",") if part.strip()]
    if len(values) != count:
        raise ValueError(f"Expected {count} weights, got {len(values)}")
    total = sum(values)
    if total <= 0:
        raise ValueError("Weights must sum to a positive value")
    return [value / total for value in values]


def _sample(records: list[dict[str, object]], count: int, rng: random.Random) -> list[dict[str, object]]:
    if count <= 0 or not records:
        return []
    if count >= len(records):
        selected = list(records)
    else:
        selected = rng.sample(records, count)
    return [dict(record) for record in selected]


def _sample_by_view(
    records: list[dict[str, object]],
    *,
    total_count: int,
    views: list[str],
    weights: list[float],
    rng: random.Random,
) -> list[dict[str, object]]:
    if total_count <= 0:
        return []
    if not views:
        return _sample(records, total_count, rng)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("source_view", ""))].append(record)

    desired = [int(total_count * weight) for weight in weights]
    while sum(desired) < total_count:
        remainders = [total_count * weight - base for weight, base in zip(weights, desired)]
        desired[max(range(len(remainders)), key=remainders.__getitem__)] += 1

    selected: list[dict[str, object]] = []
    shortfall = 0
    for view, count in zip(views, desired):
        group = grouped.get(view, [])
        take = min(count, len(group))
        shortfall += count - take
        selected.extend(_sample(group, take, rng))

    if shortfall > 0:
        selected_ids = {id(record) for record in selected}
        leftovers = [
            record
            for view in views
            for record in grouped.get(view, [])
            if id(record) not in selected_ids
        ]
        selected.extend(_sample(leftovers, shortfall, rng))
    return selected


def _tag(records: list[dict[str, object]], role: str) -> list[dict[str, object]]:
    tagged = []
    for record in records:
        copy = dict(record)
        copy["mixture_role"] = role
        tagged.append(copy)
    return tagged


def _summarize(records: list[dict[str, object]]) -> dict[str, object]:
    token_lengths = [len(record.get("input_ids", [])) for record in records]
    boundary_positive = 0
    boundary_total = 0
    for record in records:
        labels = record.get("boundary_labels") or []
        mask = record.get("boundary_loss_mask") or [1] * len(labels)
        for label, keep in zip(labels, mask):
            if int(keep):
                boundary_total += 1
                boundary_positive += int(label)
    return {
        "num_rows": len(records),
        "source_view_counts": dict(Counter(str(record.get("source_view", "")) for record in records)),
        "source_domain_counts": dict(Counter(str(record.get("source_domain", "")) for record in records)),
        "mixture_role_counts": dict(Counter(str(record.get("mixture_role", "")) for record in records)),
        "avg_tokens": (sum(token_lengths) / len(token_lengths)) if token_lengths else 0.0,
        "min_tokens": min(token_lengths) if token_lengths else 0,
        "max_tokens": max(token_lengths) if token_lengths else 0,
        "boundary_positive_rate": (boundary_positive / boundary_total) if boundary_total else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mix semantic boundary JSONL datasets deterministically.")
    parser.add_argument("--primary_dir", required=True)
    parser.add_argument("--aux_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--primary_train_take", type=int, default=12000)
    parser.add_argument("--aux_train_take", type=int, default=2500)
    parser.add_argument("--primary_valid_take", type=int, default=2000)
    parser.add_argument("--aux_valid_take", type=int, default=200)
    parser.add_argument(
        "--primary_views",
        default="codesearchnet_humaneval_bridge,codesearchnet_mbpp_bridge,contest_description_only",
    )
    parser.add_argument("--primary_view_weights", default="0.45,0.45,0.10")
    parser.add_argument(
        "--valid_primary_views",
        default="codesearchnet_humaneval_bridge,codesearchnet_mbpp_bridge",
    )
    parser.add_argument("--valid_primary_view_weights", default="0.50,0.50")
    parser.add_argument("--seed", type=int, default=20260509)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    primary_dir = Path(args.primary_dir)
    aux_dir = Path(args.aux_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_train = _read_jsonl(primary_dir / "train.jsonl")
    primary_valid = _read_jsonl(primary_dir / "valid.jsonl")
    aux_train = _read_jsonl(aux_dir / "train.jsonl")
    aux_valid = _read_jsonl(aux_dir / "valid.jsonl")

    primary_views = _parse_csv(args.primary_views)
    valid_primary_views = _parse_csv(args.valid_primary_views)
    train_records = _tag(
        _sample_by_view(
            primary_train,
            total_count=args.primary_train_take,
            views=primary_views,
            weights=_parse_weights(args.primary_view_weights, len(primary_views)),
            rng=rng,
        ),
        "primary",
    )
    train_records.extend(_tag(_sample(aux_train, args.aux_train_take, rng), "aux"))

    valid_records = _tag(
        _sample_by_view(
            primary_valid,
            total_count=args.primary_valid_take,
            views=valid_primary_views,
            weights=_parse_weights(args.valid_primary_view_weights, len(valid_primary_views)),
            rng=rng,
        ),
        "primary",
    )
    valid_records.extend(_tag(_sample(aux_valid, args.aux_valid_take, rng), "aux"))

    rng.shuffle(train_records)
    rng.shuffle(valid_records)
    _write_jsonl(output_dir / "train.jsonl", train_records)
    _write_jsonl(output_dir / "valid.jsonl", valid_records)

    summary = {
        "seed": args.seed,
        "primary_dir": str(primary_dir),
        "aux_dir": str(aux_dir),
        "train": _summarize(train_records),
        "valid": _summarize(valid_records),
        "args": vars(args),
    }
    with (output_dir / "metadata_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
