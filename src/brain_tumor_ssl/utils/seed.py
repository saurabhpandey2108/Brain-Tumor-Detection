"""Reproducibility helpers: a single seed utility for python/numpy/torch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs.

    Args:
        seed: The integer seed to apply across all RNGs.
        deterministic: If True, also force deterministic cuDNN behaviour
            (slower, but bit-reproducible on GPU).

    Returns:
        The seed that was applied (for convenient logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
