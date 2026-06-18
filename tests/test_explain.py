"""Tests for ViT attention-rollout explainability."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from brain_tumor_ssl.config import ModelConfig
from brain_tumor_ssl.evaluation.explain import (
    attention_rollout,
    explain,
    overlay_heatmap,
)
from brain_tumor_ssl.models.classifier import build_classifier

BACKBONE = "vit_tiny_patch16_224"
IMAGE_SIZE = 64
NUM_CLASSES = 4
BATCH = 2
GRID = IMAGE_SIZE // 16  # patch16 -> 4x4 patch grid


def _classifier() -> object:
    cfg = ModelConfig(backbone=BACKBONE, pretrained=False, num_classes=NUM_CLASSES)
    return build_classifier(cfg, image_size=IMAGE_SIZE)


def test_attention_rollout_shape_and_range() -> None:
    clf = _classifier()
    images = torch.randn(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)
    heatmaps = attention_rollout(clf.backbone, images)
    assert heatmaps.shape == (BATCH, GRID, GRID)
    assert float(heatmaps.min()) >= 0.0
    assert float(heatmaps.max()) == pytest.approx(1.0)


def test_explain_returns_predictions_and_heatmaps() -> None:
    clf = _classifier()
    images = torch.randn(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)
    preds, heatmaps = explain(clf, images)
    assert preds.shape == (BATCH,)
    assert preds.min() >= 0 and preds.max() < NUM_CLASSES
    assert heatmaps.shape == (BATCH, GRID, GRID)
    # Predictions match a plain forward pass (rollout does not perturb the model).
    clf.eval()
    assert torch.equal(preds, clf(images).argmax(dim=1))


def test_head_fusion_modes_run() -> None:
    clf = _classifier()
    images = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    for fusion in ("mean", "max", "min"):
        heatmaps = attention_rollout(clf.backbone, images, head_fusion=fusion)
        assert heatmaps.shape == (1, GRID, GRID)


def test_non_vit_backbone_raises() -> None:
    mlp = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 4))
    with pytest.raises(ValueError, match="Vision Transformer"):
        attention_rollout(mlp, torch.randn(1, 3, 8, 8))


def test_overlay_heatmap_matches_image_resolution() -> None:
    image = torch.rand(3, IMAGE_SIZE, IMAGE_SIZE)
    heatmap = torch.rand(GRID, GRID)
    overlay = overlay_heatmap(image, heatmap, alpha=0.5)
    assert overlay.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert overlay.dtype == np.uint8
