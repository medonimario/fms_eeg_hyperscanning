"""Small modality-specific projection heads; no foundation-model dependencies."""

from __future__ import annotations

import torch
from torch import nn


class LinearHeads(nn.Module):
    """Map fixed EEG and movement features into a shared embedding dimension."""

    def __init__(self, eeg_dim: int, movement_dim: int, embedding_dim: int = 16) -> None:
        """Initialize independent bias-free linear maps for both modalities."""
        super().__init__()
        if min(eeg_dim, movement_dim, embedding_dim) < 1:
            raise ValueError("Feature dimensions must be positive")
        self.eeg = nn.Linear(eeg_dim, embedding_dim, bias=False)
        self.movement = nn.Linear(movement_dim, embedding_dim, bias=False)

    def forward(self, eeg: torch.Tensor, movement: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project both modalities; cosine normalization belongs to the loss/scorer."""
        return self.eeg(eeg), self.movement(movement)
