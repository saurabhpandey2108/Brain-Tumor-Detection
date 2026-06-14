"""Training callbacks: best-checkpoint tracking and early stopping by val macro-F1."""

from __future__ import annotations

import copy

import torch.nn as nn

from brain_tumor_ssl.utils.logging import get_logger

logger = get_logger()


class EarlyStopper:
    """Track the best validation macro-F1 and signal when to stop early.

    Keeps a CPU copy of the best model weights so the run can restore them after
    training, regardless of later degradation.
    """

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        """Initialise the early stopper.

        Args:
            patience: Number of epochs without improvement before stopping.
            min_delta: Minimum increase in the metric to count as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = float("-inf")
        self.best_epoch = -1
        self.best_state: dict[str, object] | None = None
        self._num_bad_epochs = 0

    def update(self, metric: float, epoch: int, model: nn.Module) -> bool:
        """Record a validation metric for an epoch.

        Args:
            metric: The validation macro-F1 (higher is better).
            epoch: Current epoch index.
            model: Model whose weights are snapshotted when the metric improves.

        Returns:
            True if this epoch is a new best, else False.
        """
        if metric > self.best_metric + self.min_delta:
            self.best_metric = metric
            self.best_epoch = epoch
            self.best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            self._num_bad_epochs = 0
            logger.info("epoch {}: new best val macro-F1 = {:.4f}", epoch, metric)
            return True
        self._num_bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        """Whether patience has been exhausted."""
        return self._num_bad_epochs >= self.patience
