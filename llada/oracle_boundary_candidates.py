from __future__ import annotations

from typing import Dict, List


def build_candidate_boundary_offsets(radius: int = 2) -> List[int]:
    radius = max(0, int(radius))
    return list(range(-radius, radius + 1))


def build_candidate_boundary_points(
    *,
    segment_token_lengths: List[int],
    boundary_index: int,
    radius: int = 2,
) -> Dict[str, object]:
    if not segment_token_lengths:
        return {
            "prior_index": None,
            "candidate_indices": [],
            "candidate_deltas": [],
        }
    max_index = max(0, len(segment_token_lengths) - 1)
    prior_index = max(0, min(int(boundary_index), max_index))
    candidate_indices: List[int] = []
    candidate_deltas: List[int] = []
    for delta in build_candidate_boundary_offsets(radius=radius):
        candidate_index = prior_index + delta
        if 0 <= candidate_index <= max_index:
            candidate_indices.append(candidate_index)
            candidate_deltas.append(delta)
    return {
        "prior_index": prior_index,
        "candidate_indices": candidate_indices,
        "candidate_deltas": candidate_deltas,
    }
