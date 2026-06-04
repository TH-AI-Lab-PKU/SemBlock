from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class TaskConditionedHeadConfig:
    hidden_size: int
    projection_size: int = 1024
    dropout: float = 0.1
    include_terminal_boundary: bool = True
    phase_label_vocab: List[str] = field(default_factory=lambda: ["preamble", "setup", "iterate", "check_update", "finalize"])
    boundary_type_vocab: List[str] = field(default_factory=list)
    phase_loss_weight: float = 0.5
    transition_loss_weight: float = 0.75
    boundary_type_loss_weight: float = 0.0
    boundary_loss_weight: float = 1.0
    phase_entropy_weight: float = 0.25
    use_boundary_type_features: bool = False


class SemanticTaskConditionedHead(nn.Module):
    def __init__(self, config: TaskConditionedHeadConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.hidden_size, config.projection_size)
        self.norm = nn.LayerNorm(config.projection_size)
        self.dropout = nn.Dropout(config.dropout)
        self.phase_head = nn.Linear(config.projection_size, len(config.phase_label_vocab))
        self.boundary_type_count = len(config.boundary_type_vocab) + 1 if config.boundary_type_vocab else 0

        transition_input_size = config.projection_size * 3
        self.transition_proj = nn.Linear(transition_input_size, config.projection_size)
        self.transition_norm = nn.LayerNorm(config.projection_size)
        self.transition_out = nn.Linear(config.projection_size, 1)

        if self.boundary_type_count:
            self.boundary_pair_proj = nn.Linear(transition_input_size, config.projection_size)
            self.boundary_pair_norm = nn.LayerNorm(config.projection_size)

            self.typed_transition_proj = nn.Linear(transition_input_size, config.projection_size)
            self.typed_transition_norm = nn.LayerNorm(config.projection_size)
            self.typed_transition_out = nn.Linear(config.projection_size, self.boundary_type_count)

            self.boundary_type_proj = nn.Linear(transition_input_size, config.projection_size)
            self.boundary_type_norm = nn.LayerNorm(config.projection_size)
            self.boundary_type_out = nn.Linear(config.projection_size, self.boundary_type_count)

            boundary_input_size = (
                config.projection_size
                + (len(config.phase_label_vocab) * 3)
                + 3
                + self.boundary_type_count
            )
        else:
            boundary_input_size = (config.projection_size * 2) + (len(config.phase_label_vocab) * 3) + 3
        self.boundary_proj = nn.Linear(boundary_input_size, config.projection_size)
        self.boundary_norm = nn.LayerNorm(config.projection_size)
        self.boundary_out = nn.Linear(config.projection_size, 1)

    def _normalize_hidden_dtype(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dtype != self.input_proj.weight.dtype:
            hidden_states = hidden_states.to(self.input_proj.weight.dtype)
        return hidden_states

    def _shift_left(self, states: torch.Tensor) -> torch.Tensor:
        return torch.cat([states[:, 1:], states[:, -1:]], dim=1)

    def _pair_features(self, shared_states: torch.Tensor) -> torch.Tensor:
        next_states = self._shift_left(shared_states)
        return torch.cat([shared_states, next_states, next_states - shared_states], dim=-1)

    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self._normalize_hidden_dtype(hidden_states)
        projected = self.input_proj(hidden_states)
        return self.dropout(self.norm(torch.tanh(projected)))

    def predict_phase_logits(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        *,
        shared_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if shared_states is None:
            if hidden_states is None:
                raise ValueError("Either hidden_states or shared_states must be provided.")
            shared_states = self.encode(hidden_states)
        return self.phase_head(shared_states)

    def predict_phase_posteriors(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        *,
        shared_states: Optional[torch.Tensor] = None,
        phase_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if phase_logits is None:
            phase_logits = self.predict_phase_logits(hidden_states, shared_states=shared_states)
        return torch.softmax(phase_logits, dim=-1)

    def _compute_phase_entropy(self, phase_posteriors: torch.Tensor) -> torch.Tensor:
        safe_posteriors = phase_posteriors.clamp_min(1e-8)
        return -(safe_posteriors * torch.log(safe_posteriors)).sum(dim=-1)

    def predict_transition_logits(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        *,
        shared_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if shared_states is None:
            if hidden_states is None:
                raise ValueError("Either hidden_states or shared_states must be provided.")
            shared_states = self.encode(hidden_states)
        pair_features = self._pair_features(shared_states)
        fused = torch.tanh(self.transition_proj(pair_features))
        fused = self.dropout(self.transition_norm(fused))
        return self.transition_out(fused).squeeze(-1)

    def predict_typed_transition_logits(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        *,
        shared_states: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not self.boundary_type_count:
            return None
        if shared_states is None:
            if hidden_states is None:
                raise ValueError("Either hidden_states or shared_states must be provided.")
            shared_states = self.encode(hidden_states)
        pair_features = self._pair_features(shared_states)
        fused = torch.tanh(self.typed_transition_proj(pair_features))
        fused = self.dropout(self.typed_transition_norm(fused))
        return self.typed_transition_out(fused)

    def predict_boundary_type_logits(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        *,
        shared_states: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not self.boundary_type_count:
            return None
        if shared_states is None:
            if hidden_states is None:
                raise ValueError("Either hidden_states or shared_states must be provided.")
            shared_states = self.encode(hidden_states)
        pair_features = self._pair_features(shared_states)
        fused = torch.tanh(self.boundary_type_proj(pair_features))
        fused = self.dropout(self.boundary_type_norm(fused))
        return self.boundary_type_out(fused)

    def predict_joint_boundary_logits(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        *,
        shared_states: Optional[torch.Tensor] = None,
        phase_posteriors: Optional[torch.Tensor] = None,
        transition_condition: Optional[torch.Tensor] = None,
        boundary_type_posteriors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if shared_states is None:
            if hidden_states is None:
                raise ValueError("Either hidden_states or shared_states must be provided.")
            shared_states = self.encode(hidden_states)
        if phase_posteriors is None:
            phase_posteriors = self.predict_phase_posteriors(shared_states=shared_states)

        next_states = self._shift_left(shared_states)
        next_phase = self._shift_left(phase_posteriors)
        delta_phase = next_phase - phase_posteriors
        phase_entropy = self._compute_phase_entropy(phase_posteriors)
        next_phase_entropy = self._shift_left(phase_entropy.unsqueeze(-1)).squeeze(-1)

        if transition_condition is None:
            transition_condition = torch.sigmoid(self.predict_transition_logits(shared_states=shared_states))
        if transition_condition.dim() == 2:
            transition_feature = transition_condition.unsqueeze(-1)
        else:
            transition_feature = transition_condition

        entropy_features = torch.stack([phase_entropy, next_phase_entropy], dim=-1)
        if self.boundary_type_count:
            pair_features = self._pair_features(shared_states)
            pair_states = torch.tanh(self.boundary_pair_proj(pair_features))
            pair_states = self.dropout(self.boundary_pair_norm(pair_states))
            if getattr(self.config, "use_boundary_type_features", False):
                if boundary_type_posteriors is None:
                    boundary_type_logits = self.predict_boundary_type_logits(shared_states=shared_states)
                    boundary_type_posteriors = torch.softmax(boundary_type_logits, dim=-1)
                boundary_type_features = boundary_type_posteriors
            else:
                boundary_type_features = shared_states.new_zeros(
                    *shared_states.shape[:2],
                    self.boundary_type_count,
                )
            if boundary_type_features is None:
                boundary_type_logits = self.predict_boundary_type_logits(shared_states=shared_states)
                boundary_type_features = torch.softmax(boundary_type_logits, dim=-1)
            boundary_features = torch.cat(
                [
                    pair_states,
                    phase_posteriors,
                    next_phase,
                    delta_phase,
                    entropy_features,
                    transition_feature,
                    boundary_type_features,
                ],
                dim=-1,
            )
        else:
            boundary_features = torch.cat(
                [
                    shared_states,
                    next_states,
                    phase_posteriors,
                    next_phase,
                    delta_phase,
                    entropy_features,
                    transition_feature,
                ],
                dim=-1,
            )
        fused = torch.tanh(self.boundary_proj(boundary_features))
        fused = self.dropout(self.boundary_norm(fused))
        return self.boundary_out(fused).squeeze(-1)

    def predict_boundary_logits(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        *,
        shared_states: Optional[torch.Tensor] = None,
        phase_posteriors: Optional[torch.Tensor] = None,
        transition_condition: Optional[torch.Tensor] = None,
        boundary_type_posteriors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.predict_joint_boundary_logits(
            hidden_states,
            shared_states=shared_states,
            phase_posteriors=phase_posteriors,
            transition_condition=transition_condition,
            boundary_type_posteriors=boundary_type_posteriors,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.predict_joint_boundary_logits(hidden_states)

    @torch.no_grad()
    def predict_boundary_proba(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.predict_joint_boundary_logits(hidden_states))

    @torch.no_grad()
    def predict_proba(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.predict_boundary_proba(hidden_states)

    @torch.no_grad()
    def predict_runtime_components(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_states = self.encode(hidden_states)
        phase_logits = self.predict_phase_logits(shared_states=shared_states)
        phase_posteriors = self.predict_phase_posteriors(shared_states=shared_states, phase_logits=phase_logits)
        transition_logits = self.predict_transition_logits(shared_states=shared_states)
        boundary_type_logits = self.predict_boundary_type_logits(shared_states=shared_states)
        boundary_type_posteriors = (
            torch.softmax(boundary_type_logits, dim=-1)
            if boundary_type_logits is not None
            else None
        )
        boundary_logits = self.predict_joint_boundary_logits(
            shared_states=shared_states,
            phase_posteriors=phase_posteriors,
            transition_condition=torch.sigmoid(transition_logits),
            boundary_type_posteriors=boundary_type_posteriors,
        )
        components = {
            "shared_states": shared_states,
            "phase_logits": phase_logits,
            "phase_posteriors": phase_posteriors,
            "transition_logits": transition_logits,
            "boundary_logits": boundary_logits,
            "phase_entropy": self._compute_phase_entropy(phase_posteriors),
        }
        typed_transition_logits = self.predict_typed_transition_logits(shared_states=shared_states)
        if typed_transition_logits is not None:
            components["typed_transition_logits"] = typed_transition_logits
        if boundary_type_logits is not None:
            components["boundary_type_logits"] = boundary_type_logits
            components["boundary_type_posteriors"] = boundary_type_posteriors
        return components

    def build_checkpoint_payload(self, metadata: Optional[Dict] = None) -> Dict:
        payload_metadata = dict(metadata or {})
        payload_metadata.setdefault("phase_label_vocab", list(self.config.phase_label_vocab))
        payload_metadata.setdefault("boundary_type_vocab", list(self.config.boundary_type_vocab))
        payload_metadata.setdefault("phase_loss_weight", float(self.config.phase_loss_weight))
        payload_metadata.setdefault("transition_loss_weight", float(self.config.transition_loss_weight))
        payload_metadata.setdefault("boundary_type_loss_weight", float(self.config.boundary_type_loss_weight))
        payload_metadata.setdefault("boundary_loss_weight", float(self.config.boundary_loss_weight))
        payload_metadata.setdefault("phase_entropy_weight", float(self.config.phase_entropy_weight))
        payload_metadata.setdefault("use_boundary_type_features", bool(self.config.use_boundary_type_features))
        return {
            "head_type": "task_conditioned_phase_boundary",
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
            "metadata": payload_metadata,
        }
