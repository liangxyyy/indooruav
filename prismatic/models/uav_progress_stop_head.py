"""Progress-aware post-action STOP classifier for IndoorUAV."""

import math
from typing import Sequence

import torch
import torch.nn as nn


class IndoorUAVProgressStopHead(nn.Module):
    """Fuse root ACT/COND states and predict several remaining-step thresholds."""

    def __init__(
        self,
        input_dim: int = 4096,
        projection_dim: int = 512,
        hidden_dim: int = 1024,
        initial_positive_rates: Sequence[float] = (0.05, 0.10, 0.15, 0.25),
    ):
        super().__init__()
        rates = tuple(float(rate) for rate in initial_positive_rates)
        if not rates or any(not 0.0 < rate < 1.0 for rate in rates):
            raise ValueError("initial_positive_rates must contain probabilities in (0,1)")

        def projector():
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, projection_dim),
                nn.GELU(),
            )

        self.action_projector = projector()
        self.condition_projector = projector()
        self.trunk = nn.Sequential(
            nn.Linear(2 * projection_dim, hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, len(rates))
        with torch.no_grad():
            self.classifier.weight.zero_()
            self.classifier.bias.copy_(
                torch.tensor([math.log(rate / (1.0 - rate)) for rate in rates])
            )

    def forward(
        self,
        root_action_hidden_state: torch.Tensor,
        root_condition_hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        if root_action_hidden_state.ndim != 2 or root_condition_hidden_state.ndim != 2:
            raise ValueError("root ACT and COND hidden states must both have shape (B,D)")
        if root_action_hidden_state.shape != root_condition_hidden_state.shape:
            raise ValueError("root ACT and COND hidden states must have matching shapes")
        action_feature = self.action_projector(root_action_hidden_state.float())
        condition_feature = self.condition_projector(root_condition_hidden_state.float())
        return self.classifier(self.trunk(torch.cat((action_feature, condition_feature), dim=-1)))
