"""timm Vision Transformer backbone factory."""

from __future__ import annotations

import timm
import torch.nn as nn

from brain_tumor_ssl.utils.logging import get_logger

logger = get_logger()


def create_backbone(
    name: str,
    pretrained: bool,
    image_size: int,
    in_chans: int = 3,
) -> tuple[nn.Module, int]:
    """Create a feature-extractor ViT backbone via timm.

    The classification head is removed (``num_classes=0``) so the module returns a
    pooled feature vector. ``image_size`` is passed to timm so the positional
    embedding matches the input resolution (enabling small images for CPU smoke
    tests).

    Args:
        name: A timm model name (e.g. ``"vit_base_patch16_224"``).
        pretrained: Whether to load ImageNet-pretrained weights.
        image_size: Input square side length in pixels.
        in_chans: Number of input channels (3; grayscale is replicated upstream).

    Returns:
        A tuple ``(backbone, feature_dim)`` where ``feature_dim`` is the pooled
        output dimensionality.
    """
    model = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=0,
        in_chans=in_chans,
        img_size=image_size,
    )
    feature_dim = int(model.num_features)
    logger.info(
        "backbone '{}' (pretrained={}, img_size={}) -> feat_dim {}",
        name,
        pretrained,
        image_size,
        feature_dim,
    )
    return model, feature_dim
