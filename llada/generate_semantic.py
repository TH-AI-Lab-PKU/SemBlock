from __future__ import annotations

import ast
import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from semantic_boundary import score_boundary_window
from semantic_runtime_carry import (
    CarryRuntimeState,
    score_candidates_with_carry,
    summarize_hidden_window,
)
from semantic_runtime_hybrid import build_delta_candidates, choose_hybrid_block_length
from semantic_runtime_length import (
    build_candidate_block_lengths,
    choose_length_only_block_length,
)
from semantic_runtime_stateful import (
    _normalize_overlap_mode,
    apply_overlap_revisit_mask,
    build_stateful_revisit_plan,
    slice_prefix_past_key_values,
)
from semantic_scheduler import (
    SchedulerTraceEvent,
    build_scheduler_trace_event,
    choose_semantic_block_length_from_scores,
)


PHASE_TRANSFER_THRESHOLD_DELTAS = {
    "normalize": 0.02,
    "init_state": 0.04,
    "enumerate_or_iterate": -0.03,
    "check_or_verify": 0.02,
    "update_state": -0.04,
    "postprocess": 0.01,
    "return_emit": 0.05,
}


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_transfer_index(
    logits: torch.Tensor,
    predicted_tokens: torch.Tensor,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    threshold: Optional[float] = None,
    threshold_by_position: Optional[torch.Tensor] = None,
):
    x0 = predicted_tokens
    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)

    if threshold is not None or threshold_by_position is not None:
        if threshold_by_position is None:
            threshold_tensor = torch.full_like(confidence, float(threshold), dtype=confidence.dtype)
        else:
            threshold_tensor = threshold_by_position.to(device=confidence.device, dtype=confidence.dtype)
            if threshold_tensor.shape != confidence.shape:
                raise ValueError("threshold_by_position must have the same shape as confidence.")
        transfer_index = mask_index & (confidence >= threshold_tensor)
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
        transfer_index = (transfer_index | force_mask) & mask_index
        return x0, transfer_index

    raise ValueError("threshold must be provided for semantic generation.")


def _normalize_runtime_cap(
    remaining_length: int,
    max_block_length: Optional[int],
) -> int:
    if max_block_length is None:
        return int(remaining_length)
    return min(int(remaining_length), max(1, int(max_block_length)))


def _build_endpoint_scores(
    scores: Sequence[float],
    candidate_block_lengths: Sequence[int],
) -> Tuple[List[int], List[float], List[int]]:
    valid_block_lengths: List[int] = []
    endpoint_scores: List[float] = []
    endpoint_positions: List[int] = []
    for block_length in candidate_block_lengths:
        block_length = int(block_length)
        boundary_index = block_length - 1
        if boundary_index < 0 or boundary_index >= len(scores):
            continue
        valid_block_lengths.append(block_length)
        endpoint_scores.append(float(scores[boundary_index]))
        endpoint_positions.append(boundary_index)
    return valid_block_lengths, endpoint_scores, endpoint_positions


def _build_candidate_summaries(
    hidden_states: torch.Tensor,
    block_start: int,
    candidate_block_lengths: Sequence[int],
) -> Optional[torch.Tensor]:
    if not candidate_block_lengths:
        return None
    summaries = [
        summarize_hidden_window(hidden_states, start=block_start, end=block_start + int(block_length))
        for block_length in candidate_block_lengths
    ]
    return torch.stack(summaries, dim=0)


def _phase_vocab(boundary_head) -> List[str]:
    config = getattr(boundary_head, "config", None)
    return list(getattr(config, "phase_label_vocab", []) or [])


def _predict_phase_ids(
    boundary_head,
    hidden_states: torch.Tensor,
) -> tuple[Optional[torch.Tensor], List[str], Optional[torch.Tensor], Optional[dict[str, torch.Tensor]]]:
    if not hasattr(boundary_head, "predict_runtime_components"):
        return None, [], None, None
    components = boundary_head.predict_runtime_components(hidden_states)
    phase_posteriors = components.get("phase_posteriors")
    if phase_posteriors is None:
        return None, [], None, components
    phase_entropy = components.get("phase_entropy")
    return torch.argmax(phase_posteriors, dim=-1), _phase_vocab(boundary_head), phase_entropy, components


def _phase_confident_mask(
    *,
    phase_entropy: Optional[torch.Tensor],
    phase_label_vocab: Sequence[str],
    phase_entropy_gate: Optional[float],
) -> Optional[torch.Tensor]:
    if phase_entropy is None or phase_entropy_gate is None or not phase_label_vocab:
        return None
    denom = max(math.log(max(len(phase_label_vocab), 2)), 1e-6)
    normalized_entropy = phase_entropy.to(torch.float32) / float(denom)
    return normalized_entropy <= float(phase_entropy_gate)


def _build_phase_thresholds(
    *,
    base_threshold: float,
    phase_ids: Optional[torch.Tensor],
    phase_label_vocab: Sequence[str],
    phase_entropy: Optional[torch.Tensor] = None,
    phase_entropy_gate: Optional[float] = None,
    enabled: bool,
) -> Optional[torch.Tensor]:
    if not enabled or phase_ids is None or not phase_label_vocab:
        return None
    thresholds = torch.full(phase_ids.shape, float(base_threshold), dtype=torch.float32, device=phase_ids.device)
    confident_mask = _phase_confident_mask(
        phase_entropy=phase_entropy,
        phase_label_vocab=phase_label_vocab,
        phase_entropy_gate=phase_entropy_gate,
    )
    for phase_id, phase_name in enumerate(phase_label_vocab):
        delta = PHASE_TRANSFER_THRESHOLD_DELTAS.get(str(phase_name), 0.0)
        if delta == 0.0:
            continue
        phase_mask = phase_ids == int(phase_id)
        if confident_mask is not None:
            phase_mask = phase_mask & confident_mask.to(device=phase_ids.device)
        thresholds = torch.where(
            phase_mask,
            torch.clamp(thresholds + float(delta), min=0.0, max=1.0),
            thresholds,
        )
    return thresholds


def _apply_phase_boundary_guard(
    *,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    phase_ids: Optional[torch.Tensor],
    phase_entropy: Optional[torch.Tensor] = None,
    phase_entropy_gate: Optional[float] = None,
    phase_label_vocab: Sequence[str] = (),
    block_start: int,
    block_end: int,
    mask_id: int,
    enabled: bool,
) -> tuple[torch.Tensor, int]:
    if not enabled or phase_ids is None:
        return mask_index, int(block_end)

    guarded = mask_index.clone()
    active_end = int(block_end)
    confident_mask = _phase_confident_mask(
        phase_entropy=phase_entropy,
        phase_label_vocab=phase_label_vocab,
        phase_entropy_gate=phase_entropy_gate,
    )
    for row_idx in range(x.shape[0]):
        masked_positions = torch.nonzero(
            x[row_idx, int(block_start):int(block_end)] == int(mask_id),
            as_tuple=False,
        ).squeeze(-1)
        if masked_positions.numel() == 0:
            continue
        first_pos = int(block_start) + int(masked_positions[0].item())
        if confident_mask is not None and not bool(confident_mask[row_idx, first_pos].item()):
            continue
        current_phase = int(phase_ids[row_idx, first_pos].item())
        row_active_end = int(block_end)
        for pos in range(first_pos + 1, int(block_end)):
            if confident_mask is not None and not bool(confident_mask[row_idx, pos].item()):
                continue
            if int(phase_ids[row_idx, pos].item()) != current_phase:
                row_active_end = pos
                break
        if row_active_end < int(block_end):
            guarded[row_idx, row_active_end:int(block_end)] = False
            active_end = min(active_end, row_active_end)
    return guarded, active_end


def _python_landing_ok(text: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: List[str] = []
    for char in text:
        if char in "([{":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return False
    if stack:
        return False
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        pass
    body = "\n".join(
        line if line.startswith((" ", "\t")) or not line.strip() else f"    {line}"
        for line in text.splitlines()
    )
    try:
        ast.parse("def _candidate():\n" + (body or "    pass"))
        return True
    except SyntaxError:
        return False


def _assess_syntax_landing(
    *,
    x: torch.Tensor,
    tokenizer,
    prompt_length: int,
    block_end: int,
    mask_id: int,
) -> dict[str, object]:
    if tokenizer is None:
        return {"syntax_landing_checked": False, "syntax_landing_ok": None}
    if (x[:, int(prompt_length):int(block_end)] == int(mask_id)).sum().item() > 0:
        return {"syntax_landing_checked": False, "syntax_landing_ok": None}
    text = tokenizer.decode(
        x[0, int(prompt_length):int(block_end)].detach().cpu().tolist(),
        skip_special_tokens=True,
    )
    return {
        "syntax_landing_checked": True,
        "syntax_landing_ok": _python_landing_ok(text),
    }


def _refine_with_prefix_cache(
    *,
    model,
    x: torch.Tensor,
    refinement_start: int,
    block_end: int,
    prefix_past_key_values,
    temperature: float,
    remasking: str,
    threshold: float,
    mask_id: int,
    phase_ids: Optional[torch.Tensor] = None,
    phase_label_vocab: Sequence[str] = (),
    phase_entropy: Optional[torch.Tensor] = None,
    phase_entropy_gate: Optional[float] = None,
    phase_aware_transfer: bool = True,
    boundary_guard: bool = True,
) -> Tuple[torch.Tensor, int]:
    nfe = 0
    while True:
        if (x[:, refinement_start:block_end] == mask_id).sum() == 0:
            break
        mask_index = (x[:, refinement_start:] == mask_id)
        mask_index[:, block_end - refinement_start:] = 0
        local_phase_ids = phase_ids[:, refinement_start:] if phase_ids is not None else None
        mask_index, _ = _apply_phase_boundary_guard(
            mask_index=mask_index,
            x=x[:, refinement_start:],
            phase_ids=local_phase_ids,
            phase_entropy=phase_entropy[:, refinement_start:] if phase_entropy is not None else None,
            phase_entropy_gate=phase_entropy_gate,
            phase_label_vocab=phase_label_vocab,
            block_start=0,
            block_end=block_end - refinement_start,
            mask_id=mask_id,
            enabled=boundary_guard,
        )
        block_output = model(
            x[:, refinement_start:],
            past_key_values=prefix_past_key_values,
            use_cache=True,
        )
        block_logits = block_output.logits
        block_logits_with_noise = add_gumbel_noise(block_logits, temperature=temperature)
        block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
        nfe += 1
        x0, transfer_index = get_transfer_index(
            logits=block_logits,
            predicted_tokens=block_predicted_tokens,
            remasking=remasking,
            mask_index=mask_index,
            x=x[:, refinement_start:],
            threshold=threshold,
            threshold_by_position=_build_phase_thresholds(
                base_threshold=threshold,
                phase_ids=local_phase_ids,
                phase_label_vocab=phase_label_vocab,
                phase_entropy=phase_entropy[:, refinement_start:] if phase_entropy is not None else None,
                phase_entropy_gate=phase_entropy_gate,
                enabled=phase_aware_transfer,
            ),
        )
        x[:, refinement_start:][transfer_index] = x0[transfer_index]
    return x, nfe


def select_semantic_block_length(
    prompt: torch.Tensor,
    gen_length: int,
    generated_length: int,
    default_block_length: int,
    boundary_head,
    hidden_states: torch.Tensor,
    boundary_threshold: Optional[float],
    boundary_window_ratio: float,
    sample_id: str,
    step_index: int,
    scheduler_variant: str = "vanilla",
    candidate_block_lengths=None,
    max_block_length: Optional[int] = None,
    carry_weight: float = 0.0,
    hybrid_deltas=None,
    scheduler_state: Optional[CarryRuntimeState] = None,
    phase_entropy_gate: Optional[float] = None,
    transition_weight: float = 0.0,
    runtime_mode: str = "boundary_only",
    runtime_components: Optional[dict[str, torch.Tensor]] = None,
) -> Tuple[int, SchedulerTraceEvent, CarryRuntimeState]:
    prompt_length = prompt.shape[1]
    block_start = prompt_length + generated_length
    remaining_length = gen_length - generated_length
    cap = min(max(int(default_block_length), 1), remaining_length)
    base_window_size = min(max(int(boundary_window_ratio * gen_length), 1), remaining_length)
    variant = str(scheduler_variant or "vanilla")
    state = scheduler_state or CarryRuntimeState()

    runtime_cap = _normalize_runtime_cap(remaining_length=remaining_length, max_block_length=max_block_length)
    runtime_default_block_length = min(cap, runtime_cap)
    trace_metadata = {
        "remaining_length": remaining_length,
        "scheduler_variant": variant,
        "runtime_mode": str(runtime_mode),
        "phase_entropy_gate": phase_entropy_gate,
        "transition_weight": float(transition_weight),
        "runtime_cap": runtime_cap,
        "carry_weight": float(carry_weight),
        "previous_gate_score": float(state.prev_gate_score),
        "previous_block_length": state.prev_block_length,
        "used_max_block_length": max_block_length is not None,
    }

    window_size = min(base_window_size, cap)
    candidate_lengths: List[int] = []
    delta_values = None
    if variant in {"length_only", "carry_gate"}:
        candidate_lengths = build_candidate_block_lengths(
            candidate_block_lengths or [runtime_default_block_length],
            remaining_length=runtime_cap,
        )
        required_window = max(candidate_lengths) if candidate_lengths else runtime_default_block_length
        window_size = min(runtime_cap, max(base_window_size, required_window))
    elif variant == "hybrid_delta":
        previous_block_length = state.prev_block_length or runtime_default_block_length
        delta_values = [int(delta) for delta in (hybrid_deltas or [0])]
        candidate_lengths = build_delta_candidates(
            previous_block_length=previous_block_length,
            deltas=delta_values,
            remaining_length=runtime_cap,
        )
        required_window = max(candidate_lengths) if candidate_lengths else runtime_default_block_length
        window_size = min(runtime_cap, max(base_window_size, required_window))
        trace_metadata["hybrid_deltas"] = list(delta_values)
    elif variant != "vanilla" and variant != "fixed":
        raise ValueError(f"Unsupported scheduler_variant: {variant}")

    scores = score_boundary_window(
        boundary_head=boundary_head,
        hidden_states=hidden_states,
        start=block_start,
        end=block_start + window_size,
    )[0].to(dtype=torch.float32).detach().cpu().tolist()
    if str(runtime_mode or "boundary_only") == "phase_conditioned" and runtime_components is not None:
        transition_logits = runtime_components.get("transition_logits")
        phase_entropy = runtime_components.get("phase_entropy")
        phase_label_vocab = _phase_vocab(boundary_head)
        if transition_logits is not None and float(transition_weight or 0.0) > 0:
            transition_scores = torch.sigmoid(
                transition_logits[:, block_start:block_start + window_size]
            )[0].to(dtype=torch.float32).detach().cpu().tolist()
            weight = max(float(transition_weight), 0.0)
            scores = [
                (float(boundary_score) + weight * float(transition_score)) / (1.0 + weight)
                for boundary_score, transition_score in zip(scores, transition_scores)
            ]
        confident_mask = _phase_confident_mask(
            phase_entropy=phase_entropy[:, block_start:block_start + window_size] if phase_entropy is not None else None,
            phase_label_vocab=phase_label_vocab,
            phase_entropy_gate=phase_entropy_gate,
        )
        if confident_mask is not None:
            confidence_values = confident_mask[0].detach().cpu().tolist()
            scores = [
                float(score) if bool(confident) else float(score) * 0.5
                for score, confident in zip(scores, confidence_values)
            ]
            trace_metadata["phase_confident_fraction"] = (
                sum(1 for value in confidence_values if bool(value)) / max(len(confidence_values), 1)
            )

    if variant == "fixed":
        selected_block_length = int(runtime_default_block_length)
        selected_boundary_index = None
        selected_score = 0.0
        next_state = CarryRuntimeState(
            prev_gate_score=selected_score,
            prev_hidden_summary=summarize_hidden_window(
                hidden_states,
                start=block_start,
                end=block_start + selected_block_length,
            ).detach(),
            prev_block_length=selected_block_length,
        )
        trace_metadata["fixed_route"] = True
    elif variant == "vanilla":
        selected_block_length, selected_boundary_index, selected_score = choose_semantic_block_length_from_scores(
            scores=scores,
            default_block_length=cap,
            threshold=boundary_threshold,
        )
        next_state = CarryRuntimeState(
            prev_gate_score=float(selected_score),
            prev_hidden_summary=summarize_hidden_window(
                hidden_states,
                start=block_start,
                end=block_start + int(selected_block_length),
            ).detach(),
            prev_block_length=int(selected_block_length),
        )
    else:
        candidate_lengths, endpoint_scores, endpoint_indices = _build_endpoint_scores(
            scores=scores,
            candidate_block_lengths=candidate_lengths,
        )
        candidate_summaries = _build_candidate_summaries(
            hidden_states=hidden_states,
            block_start=block_start,
            candidate_block_lengths=candidate_lengths,
        )
        adjusted_scores = list(endpoint_scores)
        if variant in {"carry_gate", "hybrid_delta"} and candidate_summaries is not None:
            adjusted_scores = score_candidates_with_carry(
                raw_scores=endpoint_scores,
                candidate_summaries=candidate_summaries,
                state=state,
                carry_weight=float(carry_weight),
            )

        if variant == "length_only":
            decision = choose_length_only_block_length(
                candidate_scores=endpoint_scores,
                candidate_block_lengths=candidate_lengths,
                default_block_length=runtime_default_block_length,
                threshold=boundary_threshold,
            )
        elif variant == "carry_gate":
            decision = choose_length_only_block_length(
                candidate_scores=adjusted_scores,
                candidate_block_lengths=candidate_lengths,
                default_block_length=runtime_default_block_length,
                threshold=boundary_threshold,
            )
        else:
            decision = choose_hybrid_block_length(
                adjusted_scores=adjusted_scores,
                candidate_block_lengths=candidate_lengths,
                default_block_length=runtime_default_block_length,
                threshold=boundary_threshold,
            )

        selected_block_length = int(decision.selected_block_length)
        selected_score = float(decision.selected_score)
        if decision.selected_boundary_index is None:
            selected_boundary_index = None
            selected_hidden_summary = summarize_hidden_window(
                hidden_states,
                start=block_start,
                end=block_start + selected_block_length,
            ).detach()
        else:
            selected_boundary_index = selected_block_length - 1
            if candidate_summaries is None:
                selected_hidden_summary = summarize_hidden_window(
                    hidden_states,
                    start=block_start,
                    end=block_start + selected_block_length,
                ).detach()
            else:
                selected_hidden_summary = candidate_summaries[decision.selected_boundary_index].detach()

        trace_metadata.update(
            {
                "candidate_block_lengths": [int(length) for length in candidate_lengths],
                "candidate_boundary_indices": [int(index) for index in endpoint_indices],
                "candidate_boundary_positions": [int(block_start + index) for index in endpoint_indices],
                "candidate_endpoint_scores": [float(score) for score in endpoint_scores],
                "adjusted_candidate_scores": [float(score) for score in adjusted_scores],
                "decision_candidate_index": decision.selected_boundary_index,
            }
        )

        next_state = CarryRuntimeState(
            prev_gate_score=selected_score,
            prev_hidden_summary=selected_hidden_summary,
            prev_block_length=selected_block_length,
        )

    trace_event = build_scheduler_trace_event(
        scheduler_name="semantic",
        sample_id=sample_id,
        step_index=step_index,
        generated_length=generated_length,
        block_start=block_start,
        default_block_length=cap,
        scores=scores,
        selected_block_length=selected_block_length,
        selected_boundary_index=selected_boundary_index,
        selected_score=selected_score,
        threshold=boundary_threshold,
        metadata=trace_metadata,
    )
    return selected_block_length, trace_event, next_state


@torch.no_grad()
def generate_semantic(
    model,
    prompt,
    steps=128,
    gen_length=128,
    init_block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
    boundary_head=None,
    boundary_threshold=None,
    boundary_window_ratio=0.25,
    sample_id="sample-0",
    scheduler_variant="vanilla",
    candidate_block_lengths=None,
    max_block_length=None,
    carry_weight=0.0,
    hybrid_deltas=None,
    stateful_overlap_tokens: int = 0,
    stateful_overlap_mode: str = "gated",
    phase_entropy_gate: Optional[float] = None,
    transition_weight: float = 0.0,
    runtime_mode: str = "boundary_only",
    phase_aware_transfer: bool = True,
    boundary_guard: bool = True,
    syntax_aware_landing: bool = False,
    commit_reopen_tokens: int = 0,
    tokenizer=None,
):
    if boundary_head is None:
        raise ValueError("boundary_head is required for independent semantic scheduling.")
    if threshold is None:
        raise ValueError("threshold must be provided for semantic generation.")
    if prompt.shape[0] != 1:
        raise ValueError("Batch size > 1 is not supported.")

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    generated_length = 0
    nfe_history: List[int] = []
    block_history: List[int] = []
    trace_events: List[SchedulerTraceEvent] = []
    block_idx = 0
    scheduler_state = CarryRuntimeState()
    stateful_overlap_tokens = max(0, int(stateful_overlap_tokens or 0))
    stateful_overlap_mode = _normalize_overlap_mode(stateful_overlap_mode)
    use_stateful_runtime = stateful_overlap_tokens > 0

    while generated_length < gen_length:
        nfe = 0
        output = model(x, output_hidden_states=True, use_cache=use_stateful_runtime)
        logits = output.logits
        hidden_states = output.hidden_states[-1]
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
        nfe += 1
        phase_ids, phase_label_vocab, phase_entropy, runtime_components = _predict_phase_ids(boundary_head, hidden_states)

        block_length, trace_event, scheduler_state = select_semantic_block_length(
            prompt=prompt,
            gen_length=gen_length,
            generated_length=generated_length,
            default_block_length=init_block_length,
            boundary_head=boundary_head,
            hidden_states=hidden_states,
            boundary_threshold=boundary_threshold,
            boundary_window_ratio=boundary_window_ratio,
            sample_id=sample_id,
            step_index=block_idx,
            scheduler_variant=scheduler_variant,
            candidate_block_lengths=candidate_block_lengths,
            max_block_length=max_block_length,
            carry_weight=carry_weight,
            hybrid_deltas=hybrid_deltas,
            scheduler_state=scheduler_state,
            phase_entropy_gate=phase_entropy_gate,
            transition_weight=transition_weight,
            runtime_mode=runtime_mode,
            runtime_components=runtime_components,
        )
        trace_events.append(trace_event)
        block_history.append(block_length)

        block_start = prompt.shape[1] + generated_length
        block_end = block_start + block_length
        generated_length += block_length
        block_idx += 1

        mask_index = (x == mask_id)
        mask_index[:, block_end:] = 0
        mask_index, active_guard_end = _apply_phase_boundary_guard(
            mask_index=mask_index,
            x=x,
            phase_ids=phase_ids,
            phase_entropy=phase_entropy,
            phase_entropy_gate=phase_entropy_gate,
            phase_label_vocab=phase_label_vocab,
            block_start=block_start,
            block_end=block_end,
            mask_id=mask_id,
            enabled=boundary_guard,
        )
        phase_thresholds = _build_phase_thresholds(
            base_threshold=float(threshold),
            phase_ids=phase_ids,
            phase_label_vocab=phase_label_vocab,
            phase_entropy=phase_entropy,
            phase_entropy_gate=phase_entropy_gate,
            enabled=phase_aware_transfer,
        )
        trace_event.metadata.update(
            {
                "phase_aware_transfer": bool(phase_aware_transfer and phase_ids is not None),
                "boundary_guard": bool(boundary_guard and phase_ids is not None),
                "boundary_guard_active_end": int(active_guard_end),
            }
        )
        x0, transfer_index = get_transfer_index(
            logits=logits,
            predicted_tokens=predicted_tokens,
            remasking=remasking,
            mask_index=mask_index,
            x=x,
            threshold=threshold,
            threshold_by_position=phase_thresholds,
        )
        x[transfer_index] = x0[transfer_index]

        refinement_start = block_start
        prefix_past_key_values = None
        stateful_metadata = {
            "stateful_enabled": bool(use_stateful_runtime),
            "stateful_requested_overlap_tokens": int(stateful_overlap_tokens),
            "stateful_overlap_mode": stateful_overlap_mode,
            "stateful_overlap_tokens": 0,
            "stateful_revisit_start": int(block_start),
            "stateful_revisit_end": int(block_start),
            "stateful_carry_gate": 0.0,
            "stateful_reopened_tokens": 0,
            "stateful_used_prefix_cache": False,
        }
        if use_stateful_runtime and getattr(output, "past_key_values", None) is not None:
            revisit_plan = build_stateful_revisit_plan(
                prompt_length=prompt.shape[1],
                block_start=block_start,
                block_end=block_end,
                overlap_tokens=stateful_overlap_tokens,
                scheduler_state=scheduler_state,
                overlap_mode=stateful_overlap_mode,
            )
            refinement_start = int(revisit_plan.revisit_start)
            prefix_past_key_values = slice_prefix_past_key_values(
                output.past_key_values,
                prefix_length=refinement_start,
            )
            reopened_tokens = 0
            if revisit_plan.overlap_tokens > 0:
                x, reopened_tokens = apply_overlap_revisit_mask(
                    x=x,
                    revisit_start=refinement_start,
                    block_start=block_start,
                    mask_id=mask_id,
                )
            stateful_metadata.update(
                {
                    "stateful_overlap_tokens": int(revisit_plan.overlap_tokens),
                    "stateful_revisit_start": refinement_start,
                    "stateful_revisit_end": int(revisit_plan.revisit_end),
                    "stateful_carry_gate": float(revisit_plan.carry_gate),
                    "stateful_reopened_tokens": int(reopened_tokens),
                    "stateful_used_prefix_cache": prefix_past_key_values is not None,
                }
            )
        trace_event.metadata.update(stateful_metadata)

        while True:
            if (x[:, refinement_start:block_end] == mask_id).sum() == 0:
                break
            if prefix_past_key_values is not None:
                x, extra_nfe = _refine_with_prefix_cache(
                    model=model,
                    x=x,
                    refinement_start=refinement_start,
                    block_end=block_end,
                    prefix_past_key_values=prefix_past_key_values,
                    temperature=temperature,
                    remasking=remasking,
                    threshold=threshold,
                    mask_id=mask_id,
                    phase_ids=phase_ids,
                    phase_label_vocab=phase_label_vocab,
                    phase_entropy=phase_entropy,
                    phase_entropy_gate=phase_entropy_gate,
                    phase_aware_transfer=phase_aware_transfer,
                    boundary_guard=boundary_guard,
                )
                nfe += extra_nfe
                break
            mask_index = (x == mask_id)
            mask_index[:, block_end:] = 0
            mask_index, active_guard_end = _apply_phase_boundary_guard(
                mask_index=mask_index,
                x=x,
                phase_ids=phase_ids,
                phase_entropy=phase_entropy,
                phase_entropy_gate=phase_entropy_gate,
                phase_label_vocab=phase_label_vocab,
                block_start=refinement_start,
                block_end=block_end,
                mask_id=mask_id,
                enabled=boundary_guard,
            )
            trace_event.metadata["boundary_guard_active_end"] = int(active_guard_end)
            block_output = model(x)
            block_logits = block_output.logits
            block_logits_with_noise = add_gumbel_noise(block_logits, temperature=temperature)
            block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
            nfe += 1
            x0, transfer_index = get_transfer_index(
                logits=block_logits,
                predicted_tokens=block_predicted_tokens,
                remasking=remasking,
                mask_index=mask_index,
                x=x,
                threshold=threshold,
                threshold_by_position=phase_thresholds,
            )
            x[transfer_index] = x0[transfer_index]

        landing_metadata = {"commit_reopened_tokens": 0}
        if syntax_aware_landing:
            landing_metadata.update(
                _assess_syntax_landing(
                    x=x,
                    tokenizer=tokenizer,
                    prompt_length=prompt.shape[1],
                    block_end=block_end,
                    mask_id=mask_id,
                )
            )
            if landing_metadata.get("syntax_landing_ok") is False and int(commit_reopen_tokens or 0) > 0:
                reopen_start = max(prompt.shape[1], block_end - int(commit_reopen_tokens))
                reopened = int((x[:, reopen_start:block_end] != int(mask_id)).sum().item())
                x[:, reopen_start:block_end] = int(mask_id)
                landing_metadata["commit_reopened_tokens"] = reopened
                while (x[:, reopen_start:block_end] == mask_id).sum() > 0:
                    mask_index = (x == mask_id)
                    mask_index[:, block_end:] = 0
                    mask_index, _ = _apply_phase_boundary_guard(
                        mask_index=mask_index,
                        x=x,
                        phase_ids=phase_ids,
                        phase_entropy=phase_entropy,
                        phase_entropy_gate=phase_entropy_gate,
                        phase_label_vocab=phase_label_vocab,
                        block_start=reopen_start,
                        block_end=block_end,
                        mask_id=mask_id,
                        enabled=boundary_guard,
                    )
                    block_output = model(x)
                    block_logits = block_output.logits
                    block_logits_with_noise = add_gumbel_noise(block_logits, temperature=temperature)
                    block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
                    nfe += 1
                    x0, transfer_index = get_transfer_index(
                        logits=block_logits,
                        predicted_tokens=block_predicted_tokens,
                        remasking=remasking,
                        mask_index=mask_index,
                        x=x,
                        threshold=threshold,
                        threshold_by_position=phase_thresholds,
                    )
                    x[transfer_index] = x0[transfer_index]
                landing_metadata.update(
                    {
                        f"post_reopen_{key}": value
                        for key, value in _assess_syntax_landing(
                            x=x,
                            tokenizer=tokenizer,
                            prompt_length=prompt.shape[1],
                            block_end=block_end,
                            mask_id=mask_id,
                        ).items()
                    }
                )
        trace_event.metadata.update(landing_metadata)
        nfe_history.append(nfe)

    return x, nfe_history, block_history, trace_events
