from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from semantic_runtime_carry import CarryRuntimeState


@dataclass
class StatefulRevisitPlan:
    revisit_start: int
    revisit_end: int
    overlap_tokens: int
    carry_gate: float


def _normalize_overlap_mode(value: str | None) -> str:
    normalized = str(value or "gated").strip().lower()
    if normalized not in {"gated", "fixed"}:
        raise ValueError(f"Unsupported stateful overlap mode: {value}")
    return normalized


def _clamp_unit_interval(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def build_stateful_revisit_plan(
    *,
    prompt_length: int,
    block_start: int,
    block_end: int,
    overlap_tokens: int,
    scheduler_state: Optional[CarryRuntimeState] = None,
    overlap_mode: str = "gated",
) -> StatefulRevisitPlan:
    prompt_end = int(prompt_length)
    block_start_idx = int(block_start)
    block_end_idx = int(block_end)
    overlap_budget = max(0, int(overlap_tokens))
    normalized_mode = _normalize_overlap_mode(overlap_mode)

    if block_start_idx <= prompt_end or overlap_budget <= 0:
        return StatefulRevisitPlan(
            revisit_start=block_start_idx,
            revisit_end=block_start_idx,
            overlap_tokens=0,
            carry_gate=0.0,
        )

    # High boundary confidence implies less need to reopen the previous tail.
    boundary_score = _clamp_unit_interval(getattr(scheduler_state, "prev_gate_score", 0.0))
    if normalized_mode == "fixed":
        carry_gate = 1.0
        gated_overlap = overlap_budget
    else:
        carry_gate = 1.0 - boundary_score
        gated_overlap = int(round(overlap_budget * carry_gate))
        if carry_gate > 0.0 and overlap_budget > 0:
            gated_overlap = max(1, gated_overlap)

    reopen_budget = min(max(0, block_start_idx - prompt_end), max(0, gated_overlap))
    revisit_start = block_start_idx - reopen_budget

    return StatefulRevisitPlan(
        revisit_start=max(prompt_end, revisit_start),
        revisit_end=min(block_start_idx, block_end_idx),
        overlap_tokens=int(reopen_budget),
        carry_gate=float(carry_gate),
    )


def apply_overlap_revisit_mask(
    *,
    x: torch.Tensor,
    revisit_start: int,
    block_start: int,
    mask_id: int,
) -> Tuple[torch.Tensor, int]:
    updated = x.clone()
    start_idx = max(0, int(revisit_start))
    end_idx = max(start_idx, int(block_start))
    if end_idx <= start_idx:
        return updated, 0

    window = updated[:, start_idx:end_idx]
    reopened = int((window != int(mask_id)).sum().item())
    updated[:, start_idx:end_idx] = int(mask_id)
    return updated, reopened


def slice_prefix_past_key_values(
    past_key_values: Optional[Sequence[Sequence[torch.Tensor]]],
    *,
    prefix_length: int,
) -> Optional[tuple[tuple[torch.Tensor, ...], ...]]:
    if past_key_values is None:
        return None

    prefix = max(0, int(prefix_length))
    sliced_layers: list[tuple[torch.Tensor, ...]] = []
    for layer in past_key_values:
        sliced_layers.append(
            tuple(tensor[:, :, :prefix].contiguous() for tensor in layer)
        )
    return tuple(sliced_layers)
