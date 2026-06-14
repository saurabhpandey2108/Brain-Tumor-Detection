"""Network heads: SimCLR projection head and a linear classifier head."""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """SimCLR projection head: a 2-layer MLP with batch norm.

    Maps backbone features to the space where NT-Xent contrastive loss is applied.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        """Initialise the projection head.

        Args:
            in_dim: Input feature dimension (backbone output).
            hidden_dim: Hidden layer width.
            out_dim: Projection (embedding) dimension.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features to the contrastive embedding space."""
        return self.net(x)


class ClassifierHead(nn.Module):
    """Linear classification head with optional dropout."""

    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.0) -> None:
        """Initialise the classifier head.

        Args:
            in_dim: Input feature dimension (backbone output).
            num_classes: Number of output classes.
            dropout: Dropout probability applied before the linear layer.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for input features."""
        return self.fc(self.dropout(x))
