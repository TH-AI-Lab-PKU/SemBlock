from __future__ import annotations

import numbers
from typing import Dict, Mapping, Sequence

import torch


def _mean_window(hidden_states: torch.Tensor, start: int, end: int) -> torch.Tensor:
    if hidden_states.dim() != 3:
        raise ValueError("hidden_states must be 3D (batch, seq_len, hidden_size)")
    batch_size, seq_len, _ = hidden_states.shape
    if batch_size != 1:
        raise ValueError("hidden_states must have batch size == 1")
    if seq_len == 0:
        raise ValueError("hidden_states must have seq_len > 0")

    start_idx = max(0, min(seq_len, int(start)))
    end_idx = max(0, min(seq_len, int(end)))

    if end_idx <= start_idx:
        if end_idx <= 0:
            token_index = 0
        elif start_idx >= seq_len:
            token_index = seq_len - 1
        else:
            token_index = min(max(start_idx, 0), seq_len - 1)
        return hidden_states[0, token_index, :]

    return hidden_states[0, start_idx:end_idx, :].mean(dim=0)


def pool_boundary_hidden_features(
    hidden_states: torch.Tensor,
    *,
    boundary_token_index: int,
    window: int = 1,
) -> Dict[str, torch.Tensor]:
    if isinstance(boundary_token_index, numbers.Integral):
        boundary_token_index = int(boundary_token_index)
    elif isinstance(boundary_token_index, numbers.Real):
        if not float(boundary_token_index).is_integer():
            raise ValueError("boundary_token_index must be an integer")
        boundary_token_index = int(boundary_token_index)
    else:
        raise ValueError("boundary_token_index must be an integer")
    window = max(1, int(window))
    left = _mean_window(hidden_states, boundary_token_index - window, boundary_token_index)
    center = _mean_window(hidden_states, boundary_token_index, boundary_token_index + 1)
    right = _mean_window(hidden_states, boundary_token_index + 1, boundary_token_index + 1 + window)
    return {"left": left, "center": center, "right": right}


def boundary_index_to_token_index(oracle_block_sizes: Sequence[int], *, boundary_index: int) -> int:
    block_sizes = [max(1, int(size)) for size in list(oracle_block_sizes or [])]
    if not block_sizes:
        raise ValueError("oracle_block_sizes must be non-empty")
    clamped_index = max(0, min(int(boundary_index), len(block_sizes) - 1))
    return max(0, sum(block_sizes[: clamped_index + 1]) - 1)


def build_boundary_feature_vector(
    *,
    hidden_states: torch.Tensor,
    oracle_block_sizes: Sequence[int],
    boundary_index: int,
    prior_boundary_point: Mapping[str, object] | None = None,
    has_final_answer_anchor: bool = False,
    window: int = 1,
) -> torch.Tensor:
    block_sizes = [max(1, int(size)) for size in list(oracle_block_sizes or [])]
    if not block_sizes:
        raise ValueError("oracle_block_sizes must be non-empty")
    clamped_index = max(0, min(int(boundary_index), len(block_sizes) - 1))
    boundary_token_index = boundary_index_to_token_index(block_sizes, boundary_index=clamped_index)
    pooled = pool_boundary_hidden_features(
        hidden_states,
        boundary_token_index=boundary_token_index,
        window=window,
    )
    left = pooled["left"]
    center = pooled["center"]
    right = pooled["right"]
    contrast = right - left

    total_tokens = max(1, sum(block_sizes))
    current_size = block_sizes[clamped_index]
    prev_size = block_sizes[clamped_index - 1] if clamped_index > 0 else 0
    next_size = block_sizes[clamped_index + 1] if clamped_index + 1 < len(block_sizes) else 0
    tokens_before = sum(block_sizes[:clamped_index])
    tokens_after = max(0, total_tokens - tokens_before - current_size)
    boundary_ratio = 0.0 if len(block_sizes) <= 1 else clamped_index / float(len(block_sizes) - 1)
    candidate_count = len(list((prior_boundary_point or {}).get("candidate_indices") or []))
    candidate_ratio = min(1.0, candidate_count / 5.0)

    structural = torch.tensor(
        [
            current_size / float(total_tokens),
            prev_size / float(total_tokens),
            next_size / float(total_tokens),
            tokens_before / float(total_tokens),
            tokens_after / float(total_tokens),
            boundary_ratio,
            candidate_ratio,
            1.0 if has_final_answer_anchor else 0.0,
        ],
        dtype=left.dtype,
        device=left.device,
    )
    return torch.cat([left, center, right, contrast, structural], dim=0)


def build_transition_feature_matrix(
    *,
    hidden_states: torch.Tensor,
    oracle_block_sizes: Sequence[int],
    prior_boundary_points: Sequence[Mapping[str, object]] | None = None,
    has_final_answer_anchor: bool = False,
    window: int = 1,
) -> torch.Tensor:
    block_sizes = [max(1, int(size)) for size in list(oracle_block_sizes or [])]
    if not block_sizes:
        raise ValueError("oracle_block_sizes must be non-empty")

    transition_count = max(0, len(block_sizes) - 1)
    feature_vectors = []
    normalized_points = list(prior_boundary_points or [])
    for transition_index in range(transition_count):
        prior_boundary_point = None
        if transition_index < len(normalized_points):
            candidate_point = normalized_points[transition_index]
            if isinstance(candidate_point, Mapping):
                prior_boundary_point = candidate_point
        feature_vectors.append(
            build_boundary_feature_vector(
                hidden_states=hidden_states,
                oracle_block_sizes=block_sizes,
                boundary_index=transition_index,
                prior_boundary_point=prior_boundary_point,
                has_final_answer_anchor=has_final_answer_anchor,
                window=window,
            ).to(dtype=torch.float32)
        )

    if not feature_vectors:
        return torch.zeros((0, 0), dtype=torch.float32)
    return torch.stack(feature_vectors, dim=0)
