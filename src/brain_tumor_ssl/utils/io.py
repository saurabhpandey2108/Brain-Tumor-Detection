"""I/O helpers: checkpoint save/load and results.csv accumulation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch

from brain_tumor_ssl.utils.logging import get_logger

logger = get_logger()


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if it does not exist.

    Args:
        path: Directory path.

    Returns:
        The same path, now guaranteed to exist.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(state: dict[str, Any], path: Path) -> Path:
    """Save a checkpoint dictionary to disk.

    Args:
        state: Serializable state (e.g. ``{"model": state_dict, "config_hash": ...}``).
        path: Destination ``.pt`` path (parent dirs are created).

    Returns:
        The path written.
    """
    ensure_dir(path.parent)
    torch.save(state, path)
    logger.info("saved checkpoint -> {}", path)
    return path


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint dictionary from disk.

    Args:
        path: Source ``.pt`` path.
        map_location: Device to map tensors onto.

    Returns:
        The loaded state dictionary.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def append_results_row(csv_path: Path, row: dict[str, Any]) -> None:
    """Append one result row to a CSV, writing the header if the file is new.

    Args:
        csv_path: Path to ``results.csv``.
        row: Mapping of column name to value for this run.
    """
    ensure_dir(csv_path.parent)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logger.info("appended result row -> {}", csv_path)
