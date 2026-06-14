"""Device resolution helpers for CPU/GPU portability (HPC-friendly)."""

from __future__ import annotations

import torch

from brain_tumor_ssl.config import Device
from brain_tumor_ssl.utils.logging import get_logger

logger = get_logger()


def resolve_device(device: Device) -> torch.device:
    """Resolve a configured device string into a concrete ``torch.device``.

    Args:
        device: ``"cpu"``, ``"cuda"`` or ``"auto"`` (auto picks CUDA if available).

    Returns:
        The resolved device.
    """
    if device == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        logger.warning("device='cuda' requested but CUDA is unavailable; falling back to CPU")
        resolved = "cpu"
    else:
        resolved = device
    logger.info("using device: {}", resolved)
    return torch.device(resolved)
