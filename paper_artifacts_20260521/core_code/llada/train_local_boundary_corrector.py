import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

try:
    from .models.local_boundary_corrector import LocalBoundaryCorrector
except ImportError:
    from models.local_boundary_corrector import LocalBoundaryCorrector


def _load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc
    if not rows:
        raise ValueError("No training rows found")
    return rows


def _validate_rows(rows, *, gate_only: bool = False):
    if not rows:
        raise ValueError("No training rows provided")
    feature_len = None
    delta_classes = 5
    required = ("features", "gate_target") if gate_only else ("features", "gate_target", "delta_target")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} must be an object")
        missing = [key for key in required if key not in row]
        if missing:
            missing_list = ", ".join(missing)
            raise ValueError(f"Row {index} missing keys: {missing_list}")
        features = row["features"]
        if not isinstance(features, (list, tuple)):
            raise ValueError(f"Row {index} features must be a non-empty vector")
        if len(features) == 0:
            raise ValueError(f"Row {index} features must be a non-empty vector")
        if feature_len is None:
            feature_len = len(features)
        elif len(features) != feature_len:
            raise ValueError(
                f"Row {index} features length mismatch: expected {feature_len}, got {len(features)}"
            )
        for feat_index, value in enumerate(features):
            if isinstance(value, bool):
                raise ValueError(
                    f"Row {index} features must be numeric and finite; found bool at {feat_index}"
                )
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Row {index} features must be numeric; found {type(value).__name__} at {feat_index}"
                )
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"Row {index} features must be finite; found {value} at {feat_index}"
                )
        gate_target = row["gate_target"]
        if isinstance(gate_target, bool) or not isinstance(gate_target, int) or gate_target not in (0, 1):
            raise ValueError(f"Row {index} gate_target must be 0 or 1")
        if not gate_only:
            delta_target = row["delta_target"]
            if (
                isinstance(delta_target, bool)
                or not isinstance(delta_target, int)
                or not (0 <= delta_target < delta_classes)
            ):
                raise ValueError(
                    f"Row {index} delta_target must be an integer in [0, {delta_classes - 1}]"
                )
    return feature_len, delta_classes


def _train(
    rows,
    output_path: Path,
    epochs: int = 3,
    batch_size: int = 32,
    *,
    gate_only: bool = False,
) -> None:
    input_dim, delta_classes = _validate_rows(rows, gate_only=gate_only)
    hidden_dim = 128
    model = LocalBoundaryCorrector(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        delta_classes=delta_classes,
    )

    features = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
    gate_targets = torch.tensor([row["gate_target"] for row in rows], dtype=torch.long)
    delta_targets = None
    if not gate_only:
        delta_targets = torch.tensor([row["delta_target"] for row in rows], dtype=torch.long)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    total = features.size(0)
    batch_size = min(batch_size, total)
    for _ in range(epochs):
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_features = features[start:end]
            batch_gate_targets = gate_targets[start:end]
            optimizer.zero_grad()
            gate_logits, delta_logits = model(batch_features)
            loss = criterion(gate_logits, batch_gate_targets)
            if not gate_only:
                batch_delta_targets = delta_targets[start:end]
                loss = loss + 0.5 * criterion(delta_logits, batch_delta_targets)
            loss.backward()
            optimizer.step()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local boundary corrector.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()

    rows = _load_rows(Path(args.train_jsonl))
    _train(rows, Path(args.output), gate_only=args.gate_only)


if __name__ == "__main__":
    main()
