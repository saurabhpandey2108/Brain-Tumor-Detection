"""Tests for classification metrics against a hand-computed example."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from brain_tumor_ssl.evaluation.metrics import compute_metrics, evaluate

CLASS_NAMES = ["a", "b", "c"]


def test_compute_metrics_matches_hand_example() -> None:
    # 6 samples; one class-0 example is misclassified as class 1.
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2])

    m = compute_metrics(y_true, y_pred, CLASS_NAMES)

    assert m.accuracy == pytest.approx(5 / 6)
    # f1: class a = 0.6667, class b = 0.8, class c = 1.0
    assert m.per_class_f1["a"] == pytest.approx(2 / 3)
    assert m.per_class_f1["b"] == pytest.approx(0.8)
    assert m.per_class_f1["c"] == pytest.approx(1.0)
    assert m.macro_f1 == pytest.approx((2 / 3 + 0.8 + 1.0) / 3)
    assert m.confusion_matrix == [[1, 1, 0], [0, 2, 0], [0, 0, 2]]
    assert "precision" in m.report


def test_perfect_prediction_scores_one() -> None:
    y = np.array([0, 1, 2, 0, 1, 2])
    m = compute_metrics(y, y, CLASS_NAMES)
    assert m.accuracy == pytest.approx(1.0)
    assert m.macro_f1 == pytest.approx(1.0)


class _ArgmaxModel(torch.nn.Module):
    """Identity model: inputs are already logits."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_evaluate_runs_over_a_loader() -> None:
    # Logits whose argmax gives predictions [0, 1, 2]; labels match.
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    labels = torch.tensor([0, 1, 2])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    m = evaluate(_ArgmaxModel(), loader, torch.device("cpu"), CLASS_NAMES)
    assert m.accuracy == pytest.approx(1.0)
