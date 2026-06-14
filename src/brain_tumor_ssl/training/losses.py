"""Self-supervised losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Normalised temperature-scaled cross-entropy (NT-Xent) loss for SimCLR.

    Given two batches of projections for the same ``B`` images under two
    augmentations, each example's positive is its counterpart in the other view;
    all other ``2B - 2`` examples are negatives.

    Args:
        z1: Projections of the first view, shape ``(B, D)``.
        z2: Projections of the second view, shape ``(B, D)``.
        temperature: Softmax temperature (> 0).

    Returns:
        Scalar NT-Xent loss.
    """
    batch_size = z1.size(0)
    features = F.normalize(torch.cat([z1, z2], dim=0), dim=1)  # (2B, D)
    similarity = features @ features.t() / temperature  # (2B, 2B)

    # Mask self-similarity so an example is never its own positive/negative.
    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
    similarity.masked_fill_(self_mask, float("-inf"))

    # Positive of index i in [0, B) is i + B, and vice versa.
    targets = torch.cat(
        [
            torch.arange(batch_size, device=features.device) + batch_size,
            torch.arange(batch_size, device=features.device),
        ]
    )
    return F.cross_entropy(similarity, targets)
