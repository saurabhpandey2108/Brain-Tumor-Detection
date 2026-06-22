"""Tests for single-image inference + explanation (the Streamlit app's logic)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from brain_tumor_ssl.config import load_config
from brain_tumor_ssl.data.synthetic import generate_synthetic_dataset
from brain_tumor_ssl.inference import Prediction, load_classifier, predict_image
from brain_tumor_ssl.runner import run_finetune

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


@pytest.fixture
def trained(tmp_path: Path):
    data_root = tmp_path / "data"
    generate_synthetic_dataset(data_root, CLASSES, per_class=8, image_size=32, seed=0)
    cfg = load_config(
        "configs/config.yaml",
        overrides={
            "data": {"root": str(data_root), "image_size": 32, "split_strategy": "phash"},
            "model": {"backbone": "vit_tiny_patch16_224", "pretrained": False},
            "finetune": {"method": "supervised", "epochs": 1, "batch_size": 4},
            "experiment": {
                "device": "cpu",
                "workers": 0,
                "label_fractions": [1.0],
                "seeds": [42],
                "output_dir": str(tmp_path / "results"),
            },
        },
    )
    ckpt = tmp_path / "clf.pt"
    run_finetune(cfg, label_fraction=1.0, seed=42, checkpoint_path=ckpt)
    return cfg, ckpt


def test_predict_returns_valid_probabilities(trained) -> None:
    cfg, ckpt = trained
    clf = load_classifier(ckpt, cfg)
    image = Image.fromarray(np.full((40, 40), 128, dtype=np.uint8), mode="L")

    pred = predict_image(clf, image, cfg, explain=False)

    assert isinstance(pred, Prediction)
    assert pred.label in CLASSES
    assert set(pred.probabilities) == set(CLASSES)
    assert pred.probabilities[pred.label] == max(pred.probabilities.values())
    assert sum(pred.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert pred.overlay is None


def test_predict_produces_overlay(trained) -> None:
    cfg, ckpt = trained
    clf = load_classifier(ckpt, cfg)
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (50, 60), dtype=np.uint8))

    pred = predict_image(clf, image, cfg, explain=True)

    assert pred.overlay is not None
    # overlay_heatmap upsamples to the (square) base image size -> (32, 32, 3) uint8.
    assert pred.overlay.shape == (cfg.data.image_size, cfg.data.image_size, 3)
    assert pred.overlay.dtype == np.uint8


def test_accepts_rgb_input(trained) -> None:
    cfg, ckpt = trained
    clf = load_classifier(ckpt, cfg)
    rgb = Image.fromarray(np.random.default_rng(1).integers(0, 256, (33, 33, 3), dtype=np.uint8))

    pred = predict_image(clf, rgb, cfg, explain=False)
    assert pred.label in CLASSES
