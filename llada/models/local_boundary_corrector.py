import torch
from torch import nn


class LocalBoundaryCorrector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, delta_classes: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gate_head = nn.Linear(hidden_dim, 2)
        self.delta_head = nn.Linear(hidden_dim, delta_classes)

    def forward(self, features: torch.Tensor):
        hidden = self.backbone(features)
        gate_logits = self.gate_head(hidden)
        delta_logits = self.delta_head(hidden)
        return gate_logits, delta_logits

