"""Shape and weight-transfer tests for the models layer."""

from __future__ import annotations

import torch

from brain_tumor_ssl.config import ModelConfig, SSLConfig
from brain_tumor_ssl.models.backbone import create_backbone
from brain_tumor_ssl.models.classifier import (
    build_classifier,
    build_simclr_model,
    load_ssl_into_classifier,
)

BACKBONE = "vit_tiny_patch16_224"
IMAGE_SIZE = 64
NUM_CLASSES = 4
BATCH = 2


def _model_cfg() -> ModelConfig:
    return ModelConfig(backbone=BACKBONE, pretrained=False, num_classes=NUM_CLASSES)


def _ssl_cfg() -> SSLConfig:
    return SSLConfig(
        epochs=1,
        lr=1e-3,
        batch_size=4,
        weight_decay=1e-6,
        temperature=0.5,
        proj_dim=16,
        proj_hidden_dim=32,
    )


def test_backbone_returns_module_and_feat_dim() -> None:
    backbone, feat_dim = create_backbone(BACKBONE, pretrained=False, image_size=IMAGE_SIZE)
    x = torch.randn(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = backbone(x)
    assert out.shape == (BATCH, feat_dim)
    assert feat_dim == 192  # vit_tiny embed dim


def test_classifier_output_shape() -> None:
    clf = build_classifier(_model_cfg(), image_size=IMAGE_SIZE)
    clf.eval()
    logits = clf(torch.randn(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert logits.shape == (BATCH, NUM_CLASSES)


def test_simclr_output_shape() -> None:
    model = build_simclr_model(_model_cfg(), _ssl_cfg(), image_size=IMAGE_SIZE)
    model.train()
    z = model(torch.randn(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert z.shape == (BATCH, _ssl_cfg().proj_dim)


def test_load_ssl_into_classifier_transfers_backbone() -> None:
    ssl_model = build_simclr_model(_model_cfg(), _ssl_cfg(), image_size=IMAGE_SIZE)
    clf = build_classifier(_model_cfg(), image_size=IMAGE_SIZE)

    # Backbones start out different (independent random init).
    key = "blocks.0.attn.qkv.weight"
    ssl_w = ssl_model.backbone.state_dict()[key]
    assert not torch.equal(clf.backbone.state_dict()[key], ssl_w)

    load_ssl_into_classifier(clf, ssl_model.state_dict())

    # After transfer the classifier's backbone matches the SSL backbone.
    assert torch.equal(clf.backbone.state_dict()[key], ssl_w)


def test_load_ssl_without_backbone_keys_raises() -> None:
    clf = build_classifier(_model_cfg(), image_size=IMAGE_SIZE)
    try:
        load_ssl_into_classifier(clf, {"projection.net.0.weight": torch.zeros(1)})
    except ValueError as exc:
        assert "backbone" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for missing backbone keys")
