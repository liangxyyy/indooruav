"""Terminal classifier for IndoorUAV root actions."""

import math

import torch
import torch.nn as nn


class IndoorUAVStopHead(nn.Module):
    """Predict whether the root action should terminate the current instruction."""

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 1024,
        initial_positive_rate: float = 0.05,
    ):
        super().__init__()
        if not 0.0 < initial_positive_rate < 1.0:
            raise ValueError("initial_positive_rate must lie in (0, 1)")
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        with torch.no_grad():
            # Start from a calibrated constant prior.  The output layer learns
            # the feature direction first; deeper layers receive gradients once
            # that direction is non-zero.
            self.network[-1].weight.zero_()
            self.network[-1].bias.fill_(
                math.log(initial_positive_rate / (1.0 - initial_positive_rate))
            )

    def forward(self, root_action_hidden_state: torch.Tensor) -> torch.Tensor:
        if root_action_hidden_state.ndim != 2:
            raise ValueError("root_action_hidden_state must have shape (B,D)")
        return self.network(root_action_hidden_state.float()).squeeze(-1)
