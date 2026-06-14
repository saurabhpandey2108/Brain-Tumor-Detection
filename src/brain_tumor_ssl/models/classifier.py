"""Composite models: SimCLR pretraining model, classifier, and SSL weight transfer."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from brain_tumor_ssl.config import ModelConfig, SSLConfig
from brain_tumor_ssl.models.backbone import create_backbone
from brain_tumor_ssl.models.heads import ClassifierHead, ProjectionHead
from brain_tumor_ssl.utils.logging import get_logger

logger = get_logger()


class SimCLRModel(nn.Module):
    """Backbone + projection head for SimCLR self-supervised pretraining."""

    def __init__(
        self, backbone: nn.Module, feature_dim: int, proj_hidden_dim: int, proj_dim: int
    ) -> None:
        """Initialise the SimCLR model.

        Args:
            backbone: Feature-extractor backbone.
            feature_dim: Backbone output dimension.
            proj_hidden_dim: Projection head hidden width.
            proj_dim: Projection (embedding) dimension.
        """
        super().__init__()
        self.backbone = backbone
        self.projection = ProjectionHead(feature_dim, proj_hidden_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the (unnormalised) projection embedding for input images."""
        return self.projection(self.backbone(x))


class Classifier(nn.Module):
    """Backbone + linear head for supervised / semi-supervised classification."""

    def __init__(
        self, backbone: nn.Module, feature_dim: int, num_classes: int, dropout: float = 0.0
    ) -> None:
        """Initialise the classifier.

        Args:
            backbone: Feature-extractor backbone.
            feature_dim: Backbone output dimension.
            num_classes: Number of output classes.
            dropout: Dropout probability before the linear head.
        """
        super().__init__()
        self.backbone = backbone
        self.head = ClassifierHead(feature_dim, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for input images."""
        return self.head(self.backbone(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled backbone features (without the classification head)."""
        return self.backbone(x)


def build_simclr_model(
    model_cfg: ModelConfig, ssl_cfg: SSLConfig, image_size: int
) -> SimCLRModel:
    """Build a :class:`SimCLRModel` from config.

    Args:
        model_cfg: Backbone configuration.
        ssl_cfg: SimCLR configuration (projection dims).
        image_size: Input square side length.

    Returns:
        An initialised SimCLR model.
    """
    backbone, feature_dim = create_backbone(model_cfg.backbone, model_cfg.pretrained, image_size)
    return SimCLRModel(backbone, feature_dim, ssl_cfg.proj_hidden_dim, ssl_cfg.proj_dim)


def build_classifier(
    model_cfg: ModelConfig, image_size: int, pretrained: bool | None = None, dropout: float = 0.0
) -> Classifier:
    """Build a :class:`Classifier` from config.

    Args:
        model_cfg: Backbone and head configuration.
        image_size: Input square side length.
        pretrained: Override for ImageNet init (defaults to ``model_cfg.pretrained``).
            Set False when SSL weights will be loaded afterwards to avoid a download.
        dropout: Dropout probability before the head.

    Returns:
        An initialised classifier.
    """
    use_pretrained = model_cfg.pretrained if pretrained is None else pretrained
    backbone, feature_dim = create_backbone(model_cfg.backbone, use_pretrained, image_size)
    return Classifier(backbone, feature_dim, model_cfg.num_classes, dropout)


def load_ssl_into_classifier(
    classifier: Classifier,
    ssl_state_dict: Mapping[str, torch.Tensor],
    strict: bool = False,
) -> Classifier:
    """Copy SimCLR backbone weights into a classifier's backbone in place.

    Projection-head weights in the SSL checkpoint are ignored; the classifier head
    keeps its fresh initialisation.

    Args:
        classifier: The classifier whose backbone is overwritten.
        ssl_state_dict: A :class:`SimCLRModel` state dict (keys like ``backbone.*``).
        strict: Whether backbone key matching must be exact.

    Returns:
        The same classifier, with SSL backbone weights loaded.
    """
    prefix = "backbone."
    backbone_state = {
        key[len(prefix) :]: value
        for key, value in ssl_state_dict.items()
        if key.startswith(prefix)
    }
    if not backbone_state:
        raise ValueError("No 'backbone.*' weights found in the SSL state dict")
    incompatible = classifier.backbone.load_state_dict(backbone_state, strict=strict)
    logger.info(
        "loaded SSL backbone: {} missing, {} unexpected keys",
        len(incompatible.missing_keys),
        len(incompatible.unexpected_keys),
    )
    return classifier
