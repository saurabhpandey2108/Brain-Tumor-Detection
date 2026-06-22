"""PyTorch datasets: a labelled set and a two-view set (SimCLR / FixMatch)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from brain_tumor_ssl.data.indexing import Sample

Transform = Callable[[Image.Image], torch.Tensor]


def load_image(path: Path) -> Image.Image:
    """Load an image as single-channel grayscale (``mode="L"``).

    Downstream transforms replicate it to 3 channels; loading as ``L`` first
    discards any spurious colour and guarantees consistent input.

    Args:
        path: Path to the image file.

    Returns:
        A PIL grayscale image.
    """
    return Image.open(path).convert("L")


class LabeledSet(Dataset[tuple[torch.Tensor, int]]):
    """A dataset yielding ``(image_tensor, label)`` pairs."""

    def __init__(self, samples: list[Sample], transform: Transform) -> None:
        """Initialise the labelled dataset.

        Args:
            samples: Labelled samples to serve.
            transform: Transform applied to each loaded image.
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return the transformed image and its integer label at ``index``."""
        sample = self.samples[index]
        image = load_image(sample.path)
        return self.transform(image), sample.label


class TwoViewSet(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A dataset yielding two augmented views of each image.

    With a single ``transform`` (random) this produces two SimCLR views. Passing a
    distinct ``transform2`` produces a FixMatch ``(weak, strong)`` pair.
    """

    def __init__(
        self,
        samples: list[Sample],
        transform: Transform,
        transform2: Transform | None = None,
    ) -> None:
        """Initialise the two-view dataset.

        Args:
            samples: Samples to serve (labels are ignored).
            transform: Transform for the first view (and second, if ``transform2``
                is None).
            transform2: Optional distinct transform for the second view.
        """
        self.samples = samples
        self.transform = transform
        self.transform2 = transform2 if transform2 is not None else transform

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two augmented views of the image at ``index``."""
        image = load_image(self.samples[index].path)
        return self.transform(image), self.transform2(image)
