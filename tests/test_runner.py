"""End-to-end tests for the run orchestration layer on synthetic data (CPU).

These exercise the same code paths the ``btssl`` CLI drives, with a tiny ViT so the
full pretrain -> finetune -> evaluate pipeline runs in seconds without a GPU.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from brain_tumor_ssl.config import load_config
from brain_tumor_ssl.data.synthetic import generate_synthetic_dataset
from brain_tumor_ssl.runner import run_evaluate, run_finetune, run_pretrain

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


@pytest.fixture
def cfg(tmp_path: Path):
    data_root = tmp_path / "data"
    generate_synthetic_dataset(data_root, CLASSES, per_class=8, image_size=32, seed=0)
    overrides = {
        "data": {"root": str(data_root), "image_size": 32, "split_strategy": "phash"},
        "model": {"backbone": "vit_tiny_patch16_224", "pretrained": False},
        "ssl": {"epochs": 1, "batch_size": 4},
        "finetune": {"method": "fixmatch", "epochs": 1, "batch_size": 4, "early_stop_patience": 2},
        "experiment": {
            "device": "cpu",
            "workers": 0,
            "label_fractions": [0.5],
            "seeds": [42],
            "output_dir": str(tmp_path / "results"),
        },
    }
    return load_config("configs/config.yaml", overrides=overrides)


def test_pretrain_writes_checkpoint(cfg) -> None:
    path = run_pretrain(cfg)
    assert path.is_file()


def test_finetune_records_a_results_row(cfg) -> None:
    ckpt = run_pretrain(cfg)
    metrics = run_finetune(cfg, label_fraction=0.5, seed=42, ssl_checkpoint=ckpt)

    assert 0.0 <= metrics.accuracy <= 1.0
    assert set(metrics.per_class_f1) == set(CLASSES)

    results_csv = cfg.experiment.output_dir / "results.csv"
    rows = list(csv.DictReader(results_csv.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["method"] == "fixmatch"
    assert rows[0]["split"] == "phash"
    assert float(rows[0]["label_fraction"]) == pytest.approx(0.5)


def test_evaluate_with_explanations(cfg, tmp_path: Path) -> None:
    clf_ckpt = tmp_path / "clf.pt"
    run_finetune(cfg, label_fraction=0.5, seed=42, checkpoint_path=clf_ckpt)
    metrics = run_evaluate(cfg, checkpoint_path=clf_ckpt, seed=42, explain_n=1)

    assert 0.0 <= metrics.accuracy <= 1.0
    overlays = list((cfg.experiment.output_dir / "explanations").glob("*.png"))
    assert len(overlays) == 1


def test_supervised_method_runs(cfg) -> None:
    sup_cfg = load_config(
        "configs/config.yaml",
        overrides={
            "data": {"root": str(cfg.data.root), "image_size": 32, "split_strategy": "phash"},
            "model": {"backbone": "vit_tiny_patch16_224", "pretrained": False},
            "finetune": {"method": "supervised", "epochs": 1, "batch_size": 4},
            "experiment": {
                "device": "cpu",
                "workers": 0,
                "label_fractions": [0.5],
                "seeds": [42],
                "output_dir": str(cfg.experiment.output_dir),
            },
        },
    )
    metrics = run_finetune(sup_cfg, label_fraction=1.0, seed=42)
    assert 0.0 <= metrics.accuracy <= 1.0
