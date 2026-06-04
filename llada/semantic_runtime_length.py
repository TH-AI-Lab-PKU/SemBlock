from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


@dataclass
class LengthSchedulerDecision:
    selected_block_length: int
    selected_boundary_index: Optional[int]
    selected_score: float
    candidate_block_lengths: List[int]


def build_candidate_block_lengths(
    candidate_block_lengths: Iterable[int], remaining_length: int
) -> List[int]:
    normalized = sorted({max(1, int(value)) for value in candidate_block_lengths})
    if remaining_length <= 0:
        return []
    capped = [min(value, int(remaining_length)) for value in normalized]
    return sorted({value for value in capped if value > 0})


def choose_length_only_block_length(
    *,
    candidate_scores: Sequence[float],
    candidate_block_lengths: Sequence[int],
    default_block_length: int,
    threshold: Optional[float],
) -> LengthSchedulerDecision:
    candidates = list(candidate_block_lengths)
    if len(candidates) != len(candidate_scores):
        raise ValueError("candidate scores and block lengths must have the same length")
    limit = threshold if threshold is not None else float("-inf")
    best_score = float("-inf")
    best_index: Optional[int] = None
    for idx, score in enumerate(candidate_scores):
        if score >= limit and score > best_score:
            best_score = score
            best_index = idx
    if best_index is None:
        fallback_score = max(candidate_scores) if candidate_scores else 0.0
        return LengthSchedulerDecision(
            selected_block_length=default_block_length,
            selected_boundary_index=None,
            selected_score=fallback_score,
            candidate_block_lengths=list(candidates),
        )
    return LengthSchedulerDecision(
        selected_block_length=candidates[best_index],
        selected_boundary_index=best_index,
        selected_score=best_score,
        candidate_block_lengths=list(candidates),
    )
