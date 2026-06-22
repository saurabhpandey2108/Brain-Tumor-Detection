"""Walk class folders into a flat, typed list of labelled samples.

The on-disk layout is ``root/<class>/.../*.{jpg,png,...}`` (the real dataset nests
one extra level, e.g. ``Dataset/glioma/glioma/*.jpg``), so indexing recurses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from brain_tumor_ssl.utils.logging import get_logger

logger = get_logger()

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
)

# Standard Kaggle brain-tumor naming: ``Tr-...`` = training, ``Te-...`` = testing.
_SOURCE_RE = re.compile(r"^(tr|te)[-_]", re.IGNORECASE)


@dataclass(frozen=True)
class Sample:
    """A single labelled image.

    Attributes:
        path: Absolute path to the image file.
        label: Integer class index (position in the configured ``classes`` list).
        class_name: Human-readable class name.
    """

    path: Path
    label: int
    class_name: str


def source_partition(sample: Sample) -> Literal["train", "test"]:
    """Infer the dataset's own train/test partition from the filename prefix.

    Args:
        sample: The sample whose filename is inspected.

    Returns:
        ``"train"`` for ``Tr-``/``Tr_`` prefixes, ``"test"`` for ``Te-``/``Te_``.
        Files without a recognised prefix default to ``"train"``.
    """
    match = _SOURCE_RE.match(sample.path.name)
    if match is None:
        return "train"
    return "train" if match.group(1).lower() == "tr" else "test"


def _load_exclude_set(exclude_list: Path | None) -> set[Path]:
    """Load a newline-delimited file of image paths to exclude.

    Args:
        exclude_list: Path to the exclude file, or None.

    Returns:
        A set of resolved paths to skip during indexing (empty if no file).
    """
    if exclude_list is None:
        return set()
    if not exclude_list.is_file():
        logger.warning("exclude_list {} not found; ignoring", exclude_list)
        return set()
    lines = exclude_list.read_text(encoding="utf-8").splitlines()
    return {Path(line.strip()).resolve() for line in lines if line.strip()}


def index_dataset(
    root: Path,
    classes: list[str],
    exclude_list: Path | None = None,
) -> list[Sample]:
    """Index a class-folder dataset into a sorted list of samples.

    Args:
        root: Dataset root containing one sub-directory per class.
        classes: Ordered class names; index in this list becomes the integer label.
        exclude_list: Optional file of image paths to skip.

    Returns:
        Deterministically sorted list of :class:`Sample`.

    Raises:
        FileNotFoundError: If ``root`` or any class sub-directory is missing.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    excluded = _load_exclude_set(exclude_list)
    samples: list[Sample] = []
    for label, class_name in enumerate(classes):
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        paths = sorted(
            p
            for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        kept = [p for p in paths if p.resolve() not in excluded]
        samples.extend(Sample(path=p, label=label, class_name=class_name) for p in kept)
        logger.debug("indexed {} images for class '{}'", len(kept), class_name)

    if not samples:
        logger.warning("no images found under {}", root)
    return sorted(samples, key=lambda s: str(s.path))
