"""Single-image inference and explanation.

CLI-free helpers used by the Streamlit app (and usable from notebooks): load a
trained classifier from a checkpoint and run one image through it to get class
probabilities plus an optional attention-rollout overlay. Mirrors the eval-time
preprocessing exactly (grayscale -> 3-channel, ImageNet normalisation) so app
predictions match what was measured during evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from brain_tumor_ssl.config import Config
from brain_tumor_ssl.data.transforms import get_transform
from brain_tumor_ssl.evaluation.explain import attention_rollout, overlay_heatmap
from brain_tumor_ssl.models.classifier import Classifier, build_classifier
from brain_tumor_ssl.utils.device import resolve_device
from brain_tumor_ssl.utils.io import load_checkpoint
from brain_tumor_ssl.utils.logging import get_logger

logger = get_logger()


@dataclass
class Prediction:
    """The result of classifying one image.

    Attributes:
        label: Predicted class name.
        label_index: Predicted class index (position in ``cfg.data.classes``).
        probabilities: Class name -> softmax probability (sums to 1).
        overlay: ``HxWx3`` uint8 attention-rollout overlay, or None if not requested.
    """

    label: str
    label_index: int
    probabilities: dict[str, float]
    overlay: np.ndarray | None


def load_classifier(
    checkpoint_path: Path, cfg: Config, device: torch.device | None = None
) -> Classifier:
    """Load a trained classifier from a checkpoint written by ``run_finetune``.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint (expects a ``"model"`` key).
        cfg: Configuration matching the trained model (backbone, classes, image size).
        device: Device to load onto (defaults to the configured device).

    Returns:
        The classifier in eval mode on ``device``.
    """
    device = device or resolve_device(cfg.experiment.device)
    state = load_checkpoint(checkpoint_path, map_location=device)
    clf = build_classifier(cfg.model, cfg.data.image_size, pretrained=False)
    clf.load_state_dict(state["model"])
    clf.to(device).eval()
    logger.info("loaded classifier from {}", checkpoint_path)
    return clf


@torch.no_grad()
def predict_image(
    clf: Classifier,
    image: Image.Image,
    cfg: Config,
    device: torch.device | None = None,
    *,
    explain: bool = True,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Prediction:
    """Classify a single PIL image and optionally build an attention overlay.

    Args:
        clf: A trained classifier (e.g. from :func:`load_classifier`).
        image: The input image (any mode; converted to grayscale like training).
        cfg: Configuration providing class names and the eval image size.
        device: Device to run on (defaults to the classifier's current device).
        explain: Whether to compute an attention-rollout overlay.
        alpha: Overlay opacity when ``explain`` is True.
        colormap: Matplotlib colormap for the overlay.

    Returns:
        A populated :class:`Prediction`.
    """
    device = device or next(clf.parameters()).device
    size = cfg.data.image_size
    gray = image.convert("L")
    tensor = get_transform("eval", size)(gray).unsqueeze(0).to(device)

    logits = clf(tensor)
    probs = F.softmax(logits, dim=1)[0].cpu()
    index = int(probs.argmax())
    probabilities = {
        name: float(score) for name, score in zip(cfg.data.classes, probs, strict=True)
    }

    overlay: np.ndarray | None = None
    if explain:
        heatmap = attention_rollout(clf.backbone, tensor)[0].cpu()
        base = np.asarray(gray.resize((size, size)))
        overlay = overlay_heatmap(base, heatmap, alpha=alpha, colormap=colormap)

    label = cfg.data.classes[index]
    logger.info("predicted '{}' (p={:.3f})", label, probabilities[label])
    return Prediction(
        label=label,
        label_index=index,
        probabilities=probabilities,
        overlay=overlay,
    )
