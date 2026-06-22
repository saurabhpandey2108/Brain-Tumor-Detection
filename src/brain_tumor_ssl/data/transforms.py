"""Grayscale-safe image transforms for MRI.

All pipelines convert to 3-channel by replicating the single grayscale channel
(``Grayscale(num_output_channels=3)``) and normalise with ImageNet statistics so a
pretrained ViT sees inputs in its expected range. Per the MRI rules, NO colour
jitter is ever used; augmentation is geometry/intensity only.
"""

from __future__ import annotations

from typing import Literal

from torchvision import transforms

TransformKind = Literal["eval", "weak", "strong", "simclr"]

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def _gaussian_kernel(image_size: int) -> int:
    """Return an odd Gaussian-blur kernel size scaled to the image size."""
    kernel = max(3, image_size // 10)
    return kernel if kernel % 2 == 1 else kernel + 1


def _normalize() -> transforms.Normalize:
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def eval_transform(image_size: int) -> transforms.Compose:
    """Deterministic transform for validation/test/inference.

    Args:
        image_size: Target square side length in pixels.

    Returns:
        A torchvision ``Compose`` producing a normalised ``(3, H, W)`` tensor.
    """
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            _normalize(),
        ]
    )


def weak_transform(image_size: int) -> transforms.Compose:
    """FixMatch weak augmentation: flip + small translation only.

    Args:
        image_size: Target square side length in pixels.

    Returns:
        A torchvision ``Compose`` producing a normalised ``(3, H, W)`` tensor.
    """
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.125, 0.125)),
            transforms.ToTensor(),
            _normalize(),
        ]
    )


def strong_transform(image_size: int) -> transforms.Compose:
    """FixMatch strong augmentation: geometry + intensity (no colour jitter).

    Args:
        image_size: Target square side length in pixels.

    Returns:
        A torchvision ``Compose`` producing a normalised ``(3, H, W)`` tensor.
    """
    kernel = _gaussian_kernel(image_size)
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomApply([transforms.GaussianBlur(kernel, sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomAutocontrast(p=0.5),
            transforms.ToTensor(),
            _normalize(),
            transforms.RandomErasing(p=0.25),
        ]
    )


def simclr_transform(image_size: int) -> transforms.Compose:
    """SimCLR view augmentation: aggressive crop + geometry/intensity (no colour jitter).

    Args:
        image_size: Target square side length in pixels.

    Returns:
        A torchvision ``Compose`` producing a normalised ``(3, H, W)`` tensor.
    """
    kernel = _gaussian_kernel(image_size)
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.RandomApply([transforms.GaussianBlur(kernel, sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomAutocontrast(p=0.5),
            transforms.ToTensor(),
            _normalize(),
        ]
    )


def get_transform(kind: TransformKind, image_size: int) -> transforms.Compose:
    """Build a transform pipeline by name.

    Args:
        kind: One of ``"eval"``, ``"weak"``, ``"strong"``, ``"simclr"``.
        image_size: Target square side length in pixels.

    Returns:
        The requested torchvision ``Compose``.

    Raises:
        ValueError: If ``kind`` is not a recognised transform name.
    """
    builders = {
        "eval": eval_transform,
        "weak": weak_transform,
        "strong": strong_transform,
        "simclr": simclr_transform,
    }
    if kind not in builders:
        raise ValueError(f"Unknown transform kind: {kind!r}")
    return builders[kind](image_size)
