"""Tests for the configuration loading/validation layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain_tumor_ssl.config import Config, ConfigError, load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def test_load_base_config() -> None:
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg, Config)
    # Sections compose correctly.
    assert cfg.data.classes == ["glioma", "meningioma", "notumor", "pituitary"]
    assert cfg.data.image_size == 224
    assert cfg.data.split_strategy in {"phash", "naive", "source"}
    assert cfg.model.num_classes == len(cfg.data.classes)
    assert cfg.finetune.method in {"supervised", "fixmatch"}
    assert cfg.finetune.fixmatch.threshold == pytest.approx(0.95)
    assert cfg.experiment.seeds == [42, 123, 456]
    assert all(0.0 < f <= 1.0 for f in cfg.experiment.label_fractions)


def test_config_hash_is_stable_and_override_sensitive() -> None:
    cfg_a = load_config(CONFIG_PATH)
    cfg_b = load_config(CONFIG_PATH)
    assert cfg_a.config_hash() == cfg_b.config_hash()

    cfg_c = load_config(CONFIG_PATH, overrides={"finetune": {"method": "supervised"}})
    assert cfg_c.finetune.method == "supervised"
    # FixMatch params survive the partial nested override (deep merge, not replace).
    assert cfg_c.finetune.fixmatch.mu == cfg_a.finetune.fixmatch.mu
    assert cfg_c.config_hash() != cfg_a.config_hash()


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ConfigError):
        load_config(CONFIG_PATH, overrides={"model": {"not_a_real_field": 1}})


def test_num_classes_mismatch_is_rejected() -> None:
    with pytest.raises(ConfigError, match="num_classes"):
        load_config(CONFIG_PATH, overrides={"model": {"num_classes": 3}})


def test_bad_label_fraction_is_rejected() -> None:
    with pytest.raises(ConfigError):
        load_config(CONFIG_PATH, overrides={"experiment": {"label_fractions": [0.0, 1.5]}})


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does_not_exist.yaml")
