from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass
class CarryRuntimeState:
    prev_gate_score: float = 0.0
    prev_hidden_summary: Optional[torch.Tensor] = None
    prev_block_length: Optional[int] = None


def summarize_hidden_window(hidden_states: torch.Tensor, start: int, end: int) -> torch.Tensor:
    if hidden_states.dim() != 3:
        raise ValueError("hidden_states must be 3D (batch, seq_len, hidden_size)")
    if hidden_states.shape[0] != 1:
        raise ValueError("hidden_states must have batch size 1")

    seq_len = int(hidden_states.shape[1])
    if seq_len <= 0:
        raise ValueError("hidden_states must contain at least one token")

    start_idx = max(0, min(int(start), seq_len - 1))
    end_idx = max(start_idx + 1, min(int(end), seq_len))

    window = hidden_states[0, start_idx:end_idx, :].to(dtype=torch.float32)
    mean_summary = window.mean(dim=0)
    last_summary = window[-1]
    return torch.cat([mean_summary, last_summary], dim=0)


def score_candidates_with_carry(
    *,
    raw_scores: Sequence[float],
    candidate_summaries: torch.Tensor,
    state: CarryRuntimeState,
    carry_weight: float,
) -> list[float]:
    scores = [float(score) for score in raw_scores]
    if not scores:
        return []

    if state.prev_hidden_summary is None or float(carry_weight) == 0.0:
        return scores

    if candidate_summaries.dim() != 2:
        raise ValueError("candidate_summaries must be 2D")

    reference = state.prev_hidden_summary.to(dtype=torch.float32, device=candidate_summaries.device)
    summaries = candidate_summaries.to(dtype=torch.float32)
    if summaries.shape[1] != reference.shape[0]:
        raise ValueError("candidate summary width must match prev_hidden_summary width")

    similarities = F.cosine_similarity(summaries, reference.unsqueeze(0), dim=1)
    gate_scale = max(0.0, float(state.prev_gate_score))
    bonus = similarities * float(carry_weight) * gate_scale
    return [score + float(delta) for score, delta in zip(scores, bonus.tolist())]
