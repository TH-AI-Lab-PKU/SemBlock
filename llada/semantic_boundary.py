from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class BoundaryHeadConfig:
    hidden_size: int
    projection_size: int = 1024
    dropout: float = 0.1
    include_terminal_boundary: bool = True


class SemanticBoundaryHead(nn.Module):
    """
    Predicts whether a semantic boundary occurs after each token.

    The head consumes contextual token states from LLaDA and scores the transition
    between token i and token i + 1. This makes it suitable for EDU boundaries,
    statement boundaries, code or markdown alternations, and proof-step changes.
    """

    def __init__(self, config: BoundaryHeadConfig):
        super().__init__()
        self.config = config
        self.left_proj = nn.Linear(config.hidden_size, config.projection_size)
        self.right_proj = nn.Linear(config.hidden_size, config.projection_size)
        self.norm = nn.LayerNorm(config.projection_size)
        self.dropout = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(config.projection_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dtype != self.left_proj.weight.dtype:
            hidden_states = hidden_states.to(self.left_proj.weight.dtype)

        next_hidden = torch.cat([hidden_states[:, 1:], hidden_states[:, -1:]], dim=1)
        fused = torch.tanh(self.left_proj(hidden_states) + self.right_proj(next_hidden))
        fused = self.dropout(self.norm(fused))
        return self.out_proj(fused).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(hidden_states))

    @torch.no_grad()
    def predict_boundary_proba(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.predict_proba(hidden_states)

    def build_checkpoint_payload(self, metadata: Optional[Dict] = None) -> Dict:
        return {
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
            "metadata": metadata or {},
        }


def infer_hidden_size(model) -> int:
    for attr in ("d_model", "hidden_size"):
        value = getattr(model.config, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError("Could not infer hidden size from the provided LLaDA model config.")


def load_boundary_head(checkpoint_path: str, device: Optional[torch.device] = None):
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    head_type = checkpoint.get("head_type", "semantic_boundary")
    if head_type == "task_conditioned_phase_boundary":
        from semantic_task_conditioned_head import SemanticTaskConditionedHead, TaskConditionedHeadConfig

        config = TaskConditionedHeadConfig(**checkpoint["config"])
        head = SemanticTaskConditionedHead(config)
    else:
        config = BoundaryHeadConfig(**checkpoint["config"])
        head = SemanticBoundaryHead(config)
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    if device is not None:
        head = head.to(device)
    return head


def save_boundary_head(
    head: SemanticBoundaryHead,
    output_path: str,
    metadata: Optional[Dict] = None,
) -> None:
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.build_checkpoint_payload(metadata=metadata), output_path)


def tokenize_segment_separator(tokenizer, segment_separator: str = "\n") -> List[int]:
    for candidate in (segment_separator, "\n", " "):
        if not candidate:
            continue
        token_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        if token_ids:
            return list(token_ids)
    return []


def build_boundary_labels_from_segments(
    tokenizer,
    segments: Sequence[str],
    include_terminal_boundary: bool = True,
    segment_separator: str = "\n",
) -> Tuple[List[int], List[int]]:
    """
    Convert a list of semantic segments into token ids and token-level boundary labels.

    Label semantics:
    - labels[i] == 1 means a semantic boundary occurs after token i.
    - labels[i] == 0 means token i is inside the current segment.
    - non-terminal segments are separated by an explicit separator token sequence
      so the token stream does not collapse segment boundaries into bare concatenation.
    """

    input_ids: List[int] = []
    boundary_labels: List[int] = []
    kept_segments = [segment for segment in segments if segment]
    separator_ids = tokenize_segment_separator(tokenizer, segment_separator=segment_separator)

    for segment_idx, segment in enumerate(kept_segments):
        segment_ids = tokenizer(segment, add_special_tokens=False)["input_ids"]
        if not segment_ids:
            continue

        input_ids.extend(segment_ids)
        boundary_labels.extend([0] * len(segment_ids))

        is_terminal = segment_idx == len(kept_segments) - 1
        if not is_terminal or include_terminal_boundary:
            boundary_labels[-1] = 1
        if not is_terminal and separator_ids:
            input_ids.extend(separator_ids)
            boundary_labels.extend([0] * len(separator_ids))

    return input_ids, boundary_labels


def build_boundary_labels_from_positions(
    input_ids: Sequence[int],
    boundary_positions: Iterable[int],
) -> List[int]:
    """
    Convert a collection of token indices into token-level labels.

    Expected convention:
    - boundary_positions contains token indices where a segment ends.
    - a boundary at position p means labels[p] == 1.
    """

    labels = [0] * len(input_ids)
    for pos in boundary_positions:
        if 0 <= int(pos) < len(labels):
            labels[int(pos)] = 1
    return labels


@torch.no_grad()
def score_boundary_window(
    boundary_head,
    hidden_states: torch.Tensor,
    start: int,
    end: int,
) -> torch.Tensor:
    """
    Returns boundary probabilities for positions in [start, end).
    """

    try:
        boundary_device = next(boundary_head.parameters()).device
    except StopIteration:
        boundary_device = hidden_states.device

    if hasattr(boundary_head, "predict_boundary_proba"):
        probs = boundary_head.predict_boundary_proba(hidden_states.to(boundary_device))
    else:
        probs = boundary_head.predict_proba(hidden_states.to(boundary_device))
    return probs[:, start:end]
