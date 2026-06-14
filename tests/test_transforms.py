"""Tests for the grayscale-safe transform pipelines."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from brain_tumor_ssl.data.transforms import get_transform

IMAGE_SIZE = 32
KINDS = ["eval", "weak", "strong", "simclr"]


def _rgb_image() -> Image.Image:
    # Distinct per-channel content to confirm grayscale conversion happens.
    return Image.new("RGB", (48, 40), color=(200, 50, 10))


@pytest.mark.parametrize("kind", KINDS)
def test_output_shape_and_dtype(kind: str) -> None:
    transform = get_transform(kind, IMAGE_SIZE)
    out = transform(_rgb_image())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("kind", KINDS)
def test_no_colour_jitter(kind: str) -> None:
    # MRI rule: never use colour jitter.
    assert "ColorJitter" not in str(get_transform(kind, IMAGE_SIZE))


def test_eval_is_deterministic() -> None:
    transform = get_transform("eval", IMAGE_SIZE)
    img = _rgb_image()
    assert torch.equal(transform(img), transform(img))


def test_simclr_views_differ() -> None:
    transform = get_transform("simclr", IMAGE_SIZE)
    img = _rgb_image()
    # Two random views of the same image should (almost surely) differ.
    assert not torch.equal(transform(img), transform(img))


def test_normalisation_is_per_channel() -> None:
    # A flat grey image is replicated across 3 channels, then per-channel ImageNet
    # normalisation makes the channels differ -> proves Normalize was applied.
    out = get_transform("eval", IMAGE_SIZE)(Image.new("L", (40, 40), color=128))
    channel_means = out.mean(dim=(1, 2))
    assert not torch.allclose(channel_means[0], channel_means[2])


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="Unknown transform kind"):
        get_transform("nope", IMAGE_SIZE)  # type: ignore[arg-type]
