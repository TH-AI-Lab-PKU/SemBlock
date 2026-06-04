from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class SchedulerTraceEvent:
    scheduler_name: str
    sample_id: str
    step_index: int
    generated_length: int
    block_start: int
    window_size: int
    default_block_length: int
    selected_block_length: int
    selected_boundary_index: Optional[int]
    selected_boundary_position: Optional[int]
    selected_score: float
    threshold: Optional[float]
    candidate_scores: List[float] = field(default_factory=list)
    candidate_positions: List[int] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def choose_semantic_block_length_from_scores(
    scores: Sequence[float],
    default_block_length: int,
    threshold: Optional[float] = None,
) -> Tuple[int, Optional[int], float]:
    fallback_length = max(1, int(default_block_length))
    if not scores:
        return fallback_length, None, 0.0

    capped_scores = [float(score) for score in scores[:fallback_length]]
    if not capped_scores:
        return fallback_length, None, 0.0

    best_index = max(range(len(capped_scores)), key=lambda idx: capped_scores[idx])
    best_score = float(capped_scores[best_index])
    if threshold is not None and best_score < float(threshold):
        return min(fallback_length, len(capped_scores)), None, best_score
    return best_index + 1, best_index, best_score


def build_scheduler_trace_event(
    scheduler_name: str,
    sample_id: str,
    step_index: int,
    generated_length: int,
    block_start: int,
    default_block_length: int,
    scores: Sequence[float],
    selected_block_length: int,
    selected_boundary_index: Optional[int],
    selected_score: float,
    threshold: Optional[float],
    metadata: Optional[Dict[str, object]] = None,
) -> SchedulerTraceEvent:
    candidate_scores = [float(score) for score in scores]
    candidate_positions = [block_start + idx for idx in range(len(candidate_scores))]
    selected_boundary_position = None
    if selected_boundary_index is not None:
        selected_boundary_position = block_start + int(selected_boundary_index)
    return SchedulerTraceEvent(
        scheduler_name=scheduler_name,
        sample_id=sample_id,
        step_index=int(step_index),
        generated_length=int(generated_length),
        block_start=int(block_start),
        window_size=len(candidate_scores),
        default_block_length=int(default_block_length),
        selected_block_length=int(selected_block_length),
        selected_boundary_index=selected_boundary_index,
        selected_boundary_position=selected_boundary_position,
        selected_score=float(selected_score),
        threshold=None if threshold is None else float(threshold),
        candidate_scores=candidate_scores,
        candidate_positions=candidate_positions,
        metadata=metadata or {},
    )


def trace_events_from_block_history(
    scheduler_name: str,
    sample_id: str,
    prompt_length: int,
    block_history: Sequence[int],
) -> List[SchedulerTraceEvent]:
    events: List[SchedulerTraceEvent] = []
    generated_length = 0
    for step_index, block_length in enumerate(block_history):
        block_length = int(block_length)
        block_start = int(prompt_length) + generated_length
        selected_boundary_index = None if block_length <= 0 else block_length - 1
        selected_boundary_position = None if block_length <= 0 else block_start + block_length - 1
        events.append(
            SchedulerTraceEvent(
                scheduler_name=scheduler_name,
                sample_id=sample_id,
                step_index=step_index,
                generated_length=generated_length,
                block_start=block_start,
                window_size=block_length,
                default_block_length=block_length,
                selected_block_length=block_length,
                selected_boundary_index=selected_boundary_index,
                selected_boundary_position=selected_boundary_position,
                selected_score=0.0,
                threshold=None,
                candidate_scores=[],
                candidate_positions=[],
                metadata={},
            )
        )
        generated_length += max(block_length, 0)
    return events


def write_scheduler_trace(path: Path, events: Iterable[SchedulerTraceEvent]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
