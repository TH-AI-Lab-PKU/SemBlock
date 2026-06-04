from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


@dataclass
class HybridSchedulerDecision:
    selected_block_length: int
    selected_boundary_index: Optional[int]
    selected_score: float
    candidate_block_lengths: List[int]


def build_delta_candidates(
    previous_block_length: int, deltas: Iterable[int], remaining_length: int
) -> List[int]:
    if remaining_length <= 0:
        return []

    remaining = int(remaining_length)
    lengths: set[int] = set()
    base = int(previous_block_length)
    for delta in deltas:
        delta_length = base + int(delta)
        delta_length = max(1, delta_length)
        delta_length = min(delta_length, remaining)
        if delta_length > 0:
            lengths.add(delta_length)

    return sorted(lengths)


def choose_hybrid_block_length(
    *,
    adjusted_scores: Sequence[float],
    candidate_block_lengths: Sequence[int],
    default_block_length: int,
    threshold: Optional[float],
) -> HybridSchedulerDecision:
    raw_scores = list(adjusted_scores)
    candidate_pairs: list[tuple[int, float]] = []
    for candidate, score in zip(candidate_block_lengths, raw_scores):
        candidate_length = int(candidate)
        if candidate_length <= 0:
            continue
        candidate_pairs.append((candidate_length, float(score)))

    if not candidate_pairs:
        return HybridSchedulerDecision(
            selected_block_length=default_block_length,
            selected_boundary_index=None,
            selected_score=0.0,
            candidate_block_lengths=[],
        )

    candidate_lengths = [length for length, _ in candidate_pairs]
    best_threshold_idx: Optional[int] = None
    best_threshold_score = float("-inf")
    best_score = float("-inf")

    for idx, (_, score) in enumerate(candidate_pairs):
        best_score = max(best_score, score)
        if threshold is None or score >= threshold:
            if score > best_threshold_score:
                best_threshold_score = score
                best_threshold_idx = idx

    selected_score = best_score if best_score != float("-inf") else 0.0

    if best_threshold_idx is not None:
        return HybridSchedulerDecision(
            selected_block_length=candidate_lengths[best_threshold_idx],
            selected_boundary_index=best_threshold_idx,
            selected_score=best_threshold_score,
            candidate_block_lengths=candidate_lengths,
        )

    return HybridSchedulerDecision(
        selected_block_length=default_block_length,
        selected_boundary_index=None,
        selected_score=selected_score,
        candidate_block_lengths=candidate_lengths,
    )
