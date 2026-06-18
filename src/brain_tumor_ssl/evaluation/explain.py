"""ViT attention-rollout explainability.

Attention rollout (Abnar & Zuidema, 2020) estimates how much each input patch
contributes to the ``[CLS]`` token's final representation. Per transformer
block the attention is averaged over heads, a residual identity term is added
(tokens always attend to themselves through the skip connection) and the rows
are renormalised; these matrices are then multiplied across all blocks. The
``[CLS]`` row of the product, reshaped to the patch grid, is an interpretable
saliency map over the image.

The attention probabilities are not exposed by timm's fused attention, so they
are recomputed from each attention module's own ``qkv`` projection inside a
forward hook during a single forward pass.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from brain_tumor_ssl.models.classifier import Classifier
from brain_tumor_ssl.utils.io import ensure_dir
from brain_tumor_ssl.utils.logging import get_logger

logger = get_logger()

HeadFusion = Literal["mean", "max", "min"]


@contextmanager
def _capture_attentions(backbone: nn.Module) -> Iterator[list[torch.Tensor]]:
    """Hook every attention block and collect per-layer attention probabilities.

    Each captured tensor has shape ``(batch, heads, tokens, tokens)`` and the
    list is ordered from the first transformer block to the last.

    Args:
        backbone: A timm Vision Transformer (attention modules expose ``qkv``,
            ``num_heads`` and ``scale``).

    Yields:
        A list that is populated with one attention tensor per block as the
        forward pass runs.
    """
    attentions: list[torch.Tensor] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: object) -> None:
        x = inputs[0]
        batch, tokens, channels = x.shape
        head_dim = channels // module.num_heads
        qkv = module.qkv(x).reshape(batch, tokens, 3, module.num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        if hasattr(module, "q_norm"):
            q = module.q_norm(q)
        if hasattr(module, "k_norm"):
            k = module.k_norm(k)
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attentions.append(attn.softmax(dim=-1).detach())

    for module in backbone.modules():
        if hasattr(module, "qkv") and hasattr(module, "num_heads"):
            handles.append(module.register_forward_hook(hook))

    try:
        yield attentions
    finally:
        for handle in handles:
            handle.remove()


def _rollout(
    attentions: list[torch.Tensor], head_fusion: HeadFusion, discard_ratio: float
) -> torch.Tensor:
    """Multiply per-layer attention matrices into a single rollout matrix.

    Args:
        attentions: Per-block attention tensors ``(batch, heads, tokens, tokens)``.
        head_fusion: How to reduce the head dimension (``mean``/``max``/``min``).
        discard_ratio: Fraction of the lowest-weight attention entries to zero out
            before rollout (noise suppression; ``0`` keeps everything).

    Returns:
        The rollout matrix ``(batch, tokens, tokens)``.
    """
    batch, _, tokens, _ = attentions[0].shape
    device = attentions[0].device
    identity = torch.eye(tokens, device=device)
    result = identity.expand(batch, tokens, tokens).clone()

    for attn in attentions:
        if head_fusion == "mean":
            fused = attn.mean(dim=1)
        elif head_fusion == "max":
            fused = attn.amax(dim=1)
        else:
            fused = attn.amin(dim=1)

        if discard_ratio > 0:
            flat = fused.reshape(batch, -1)
            num_drop = int(flat.shape[-1] * discard_ratio)
            if num_drop > 0:
                _, lowest = flat.topk(num_drop, dim=-1, largest=False)
                flat.scatter_(-1, lowest, 0.0)
            fused = flat.reshape(batch, tokens, tokens)

        # Account for the residual connection, then renormalise rows to sum to 1.
        fused = fused + identity
        fused = fused / fused.sum(dim=-1, keepdim=True)
        result = fused @ result

    return result


def _grid_shape(backbone: nn.Module, num_patches: int) -> tuple[int, int]:
    """Return the ``(height, width)`` patch grid, falling back to a square."""
    grid = getattr(getattr(backbone, "patch_embed", None), "grid_size", None)
    if isinstance(grid, (tuple, list)) and len(grid) == 2:
        return int(grid[0]), int(grid[1])
    side = round(num_patches**0.5)
    return side, side


@torch.no_grad()
def attention_rollout(
    backbone: nn.Module,
    images: torch.Tensor,
    *,
    head_fusion: HeadFusion = "mean",
    discard_ratio: float = 0.0,
) -> torch.Tensor:
    """Compute attention-rollout saliency maps for a batch of images.

    Args:
        backbone: A timm Vision Transformer backbone (e.g. ``classifier.backbone``).
        images: Input batch ``(batch, channels, H, W)``.
        head_fusion: How to reduce attention heads (``mean``/``max``/``min``).
        discard_ratio: Fraction of lowest attention entries to drop before rollout.

    Returns:
        Per-image heatmaps ``(batch, grid_h, grid_w)`` normalised to ``[0, 1]``.

    Raises:
        ValueError: If the backbone exposes no attention modules (not a ViT).
    """
    backbone.eval()
    with _capture_attentions(backbone) as attentions:
        backbone(images)
    if not attentions:
        raise ValueError("No attention modules found; is this backbone a Vision Transformer?")

    result = _rollout(attentions, head_fusion, discard_ratio)
    num_prefix = int(getattr(backbone, "num_prefix_tokens", 1))
    # Saliency of each patch as seen from the class token (row 0).
    mask = result[:, 0, num_prefix:] if num_prefix else result.mean(dim=1)

    batch = mask.shape[0]
    grid_h, grid_w = _grid_shape(backbone, mask.shape[-1])
    mask = mask.reshape(batch, grid_h, grid_w)

    flat = mask.reshape(batch, -1)
    lo = flat.min(dim=-1, keepdim=True).values
    hi = flat.max(dim=-1, keepdim=True).values
    normalised = (flat - lo) / (hi - lo).clamp_min(1e-8)
    return normalised.reshape(batch, grid_h, grid_w)


@torch.no_grad()
def explain(
    classifier: Classifier,
    images: torch.Tensor,
    *,
    head_fusion: HeadFusion = "mean",
    discard_ratio: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify a batch and produce an attention-rollout heatmap per image.

    Both the predictions and the heatmaps come from a single forward pass.

    Args:
        classifier: A trained :class:`Classifier` with a ViT backbone.
        images: Input batch ``(batch, channels, H, W)``.
        head_fusion: How to reduce attention heads (``mean``/``max``/``min``).
        discard_ratio: Fraction of lowest attention entries to drop before rollout.

    Returns:
        ``(predictions, heatmaps)`` — integer class indices ``(batch,)`` and
        ``(batch, grid_h, grid_w)`` heatmaps normalised to ``[0, 1]``.
    """
    classifier.eval()
    with _capture_attentions(classifier.backbone) as attentions:
        logits = classifier(images)
    if not attentions:
        raise ValueError("No attention modules found; is this backbone a Vision Transformer?")

    predictions = logits.argmax(dim=1)
    result = _rollout(attentions, head_fusion, discard_ratio)
    num_prefix = int(getattr(classifier.backbone, "num_prefix_tokens", 1))
    mask = result[:, 0, num_prefix:] if num_prefix else result.mean(dim=1)

    batch = mask.shape[0]
    grid_h, grid_w = _grid_shape(classifier.backbone, mask.shape[-1])
    flat = mask.reshape(batch, -1)
    lo = flat.min(dim=-1, keepdim=True).values
    hi = flat.max(dim=-1, keepdim=True).values
    heatmaps = ((flat - lo) / (hi - lo).clamp_min(1e-8)).reshape(batch, grid_h, grid_w)
    return predictions, heatmaps


def _image_to_rgb(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a CHW/HWC/HW image (tensor or array) to an ``HxWx3`` uint8 array."""
    array = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3):  # CHW -> HWC
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:  # grayscale -> RGB
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.dtype != np.uint8:
        lo, hi = float(array.min()), float(array.max())
        array = (array - lo) / (hi - lo) if hi > lo else np.zeros_like(array)
        array = (array * 255).astype(np.uint8)
    return array


def overlay_heatmap(
    image: torch.Tensor | np.ndarray,
    heatmap: torch.Tensor | np.ndarray,
    *,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """Blend a low-resolution heatmap over an image as an RGB overlay.

    Args:
        image: Source image (CHW/HWC/HW, tensor or array, any scale).
        heatmap: ``(grid_h, grid_w)`` saliency map in ``[0, 1]``.
        alpha: Heatmap opacity in ``[0, 1]`` (``0`` = image only).
        colormap: Any matplotlib colormap name.

    Returns:
        An ``HxWx3`` uint8 RGB overlay at the image's resolution.
    """
    from matplotlib import colormaps

    rgb = _image_to_rgb(image)
    height, width = rgb.shape[:2]

    hm = heatmap.detach().cpu() if isinstance(heatmap, torch.Tensor) else torch.as_tensor(heatmap)
    hm = F.interpolate(
        hm.reshape(1, 1, *hm.shape).float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).reshape(height, width)

    coloured = colormaps[colormap](hm.numpy())[..., :3]  # drop alpha channel
    coloured = (coloured * 255).astype(np.uint8)
    blended = (1 - alpha) * rgb + alpha * coloured
    return blended.clip(0, 255).astype(np.uint8)


def save_explanation(
    image: torch.Tensor | np.ndarray,
    heatmap: torch.Tensor | np.ndarray,
    path: Path,
    *,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Path:
    """Render an attention overlay and write it to ``path`` as an image.

    Args:
        image: Source image (CHW/HWC/HW, tensor or array).
        heatmap: ``(grid_h, grid_w)`` saliency map in ``[0, 1]``.
        path: Destination image path (parent directories are created).
        alpha: Heatmap opacity.
        colormap: Any matplotlib colormap name.

    Returns:
        The path written.
    """
    from PIL import Image

    overlay = overlay_heatmap(image, heatmap, alpha=alpha, colormap=colormap)
    ensure_dir(path.parent)
    Image.fromarray(overlay).save(path)
    logger.info("saved attention overlay -> {}", path)
    return path
