"""Small trainable adapter for IndoorUAV image roles and condition matching."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MatchingProjector(nn.Module):
    """Project LLM-width tokens into a normalized visual matching space."""

    def __init__(self, input_dim: int = 4096, hidden_dim: int = 1024, output_dim: int = 512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(features.float()), dim=-1)


class IndoorUAVConditionAdapter(nn.Module):
    """Own all non-OpenVLA parameters needed by visual condition matching."""

    NUM_IMAGE_ROLES = 3

    def __init__(self, llm_dim: int = 4096, hidden_dim: int = 1024, match_dim: int = 512):
        super().__init__()
        self.role_embeddings = nn.Embedding(self.NUM_IMAGE_ROLES, llm_dim)
        self.condition_projector = MatchingProjector(llm_dim, hidden_dim, match_dim)
        self.vision_projector = MatchingProjector(llm_dim, hidden_dim, match_dim)
        nn.init.normal_(self.role_embeddings.weight, mean=0.0, std=0.02)

    @property
    def image_role_embeddings(self) -> torch.Tensor:
        return self.role_embeddings.weight

    def project_conditions(self, features: torch.Tensor) -> torch.Tensor:
        return self.condition_projector(features)

    def project_vision(self, features: torch.Tensor) -> torch.Tensor:
        return self.vision_projector(features)
