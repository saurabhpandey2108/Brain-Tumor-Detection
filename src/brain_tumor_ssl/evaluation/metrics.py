"""Classification metrics: accuracy, macro-F1, per-class report, confusion matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from brain_tumor_ssl.utils.logging import get_logger

logger = get_logger()


@dataclass
class ClassificationMetrics:
    """Container for classification evaluation results.

    Attributes:
        accuracy: Overall accuracy in ``[0, 1]``.
        macro_f1: Unweighted mean per-class F1.
        per_class_f1: Mapping of class name to its F1 score.
        confusion_matrix: Confusion matrix as nested lists (rows = true labels).
        report: Human-readable sklearn classification report.
    """

    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]
    report: str


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> ClassificationMetrics:
    """Compute classification metrics from true/predicted labels.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        class_names: Class names indexed by label.

    Returns:
        A populated :class:`ClassificationMetrics`.
    """
    labels = list(range(len(class_names)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    per_class_f1 = {
        name: float(score) for name, score in zip(class_names, per_class, strict=True)
    }
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(macro),
        per_class_f1=per_class_f1,
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        report=classification_report(
            y_true, y_pred, labels=labels, target_names=class_names, zero_division=0
        ),
    )


@torch.no_grad()
def predict(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over a loader, returning true and predicted labels.

    Args:
        model: Trained classifier returning logits.
        loader: DataLoader yielding ``(images, labels)``.
        device: Device to run on.

    Returns:
        ``(y_true, y_pred)`` integer label arrays.
    """
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in loader:
        logits = model(images.to(device))
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        y_true.extend(labels.tolist())
    return np.array(y_true), np.array(y_pred)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> ClassificationMetrics:
    """Evaluate a classifier on a loader and return metrics.

    Args:
        model: Trained classifier returning logits.
        loader: DataLoader yielding ``(images, labels)``.
        device: Device to run on.
        class_names: Class names indexed by label.

    Returns:
        Computed :class:`ClassificationMetrics`.
    """
    y_true, y_pred = predict(model, loader, device)
    return compute_metrics(y_true, y_pred, class_names)
