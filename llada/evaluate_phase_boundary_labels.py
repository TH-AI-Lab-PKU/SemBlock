from __future__ import annotations

from typing import Dict, Iterable, List, Sequence


def _flatten_masked(
    values: Sequence[Sequence[int]],
    masks: Sequence[Sequence[int]],
) -> List[int]:
    flattened: List[int] = []
    for row_values, row_mask in zip(values, masks):
        for value, mask in zip(row_values, row_mask):
            if int(mask):
                flattened.append(int(value))
    return flattened


def _binary_metrics(predictions: Iterable[int], targets: Iterable[int]) -> Dict[str, float]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for prediction, target in zip(predictions, targets):
        if int(prediction) == 1 and int(target) == 1:
            true_positive += 1
        elif int(prediction) == 1 and int(target) == 0:
            false_positive += 1
        elif int(prediction) == 0 and int(target) == 1:
            false_negative += 1

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
    }


def compute_label_metrics(
    *,
    phase_predictions: Sequence[Sequence[int]],
    phase_targets: Sequence[Sequence[int]],
    phase_masks: Sequence[Sequence[int]],
    transition_predictions: Sequence[Sequence[int]] | None = None,
    transition_targets: Sequence[Sequence[int]] | None = None,
    transition_masks: Sequence[Sequence[int]] | None = None,
    boundary_predictions: Sequence[Sequence[int]],
    boundary_targets: Sequence[Sequence[int]],
    boundary_masks: Sequence[Sequence[int]],
    phase_label_vocab: Sequence[str],
) -> Dict[str, object]:
    flat_phase_predictions = _flatten_masked(phase_predictions, phase_masks)
    flat_phase_targets = _flatten_masked(phase_targets, phase_masks)

    per_label_f1: Dict[str, float] = {}
    phase_f1_values: List[float] = []
    for label_id, label_name in enumerate(phase_label_vocab):
        binary_predictions = [1 if value == label_id else 0 for value in flat_phase_predictions]
        binary_targets = [1 if value == label_id else 0 for value in flat_phase_targets]
        metrics = _binary_metrics(binary_predictions, binary_targets)
        per_label_f1[str(label_name)] = metrics["f1"]
        phase_f1_values.append(metrics["f1"])

    flat_boundary_predictions = _flatten_masked(boundary_predictions, boundary_masks)
    flat_boundary_targets = _flatten_masked(boundary_targets, boundary_masks)
    boundary_metrics = _binary_metrics(flat_boundary_predictions, flat_boundary_targets)
    if transition_predictions is None:
        transition_predictions = [[0]]
    if transition_targets is None:
        transition_targets = [[0]]
    if transition_masks is None:
        transition_masks = [[0]]
    flat_transition_predictions = _flatten_masked(transition_predictions, transition_masks)
    flat_transition_targets = _flatten_masked(transition_targets, transition_masks)
    transition_metrics = _binary_metrics(flat_transition_predictions, flat_transition_targets)

    return {
        "phase_macro_f1": 0.0 if not phase_f1_values else sum(phase_f1_values) / len(phase_f1_values),
        "phase_per_label_f1": per_label_f1,
        "transition_precision": transition_metrics["precision"],
        "transition_recall": transition_metrics["recall"],
        "transition_f1": transition_metrics["f1"],
        "boundary_precision": boundary_metrics["precision"],
        "boundary_recall": boundary_metrics["recall"],
        "boundary_f1": boundary_metrics["f1"],
    }
