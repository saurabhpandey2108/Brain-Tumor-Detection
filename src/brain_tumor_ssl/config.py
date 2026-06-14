"""Typed, validated configuration for brain-tumor-ssl.

The configuration is split across small per-concern YAML files in ``configs/``
that are composed by ``configs/config.yaml``. :func:`load_config` reads the base
file, resolves its ``includes`` into the matching sections, applies optional
runtime overrides, and validates everything with pydantic so the program fails
fast and loudly on a bad config.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

SplitStrategy = Literal["phash", "naive", "source"]
FinetuneMethod = Literal["supervised", "fixmatch"]
Device = Literal["cpu", "cuda", "auto"]


class ConfigError(RuntimeError):
    """Raised when configuration files are missing, malformed, or invalid."""


class _Strict(BaseModel):
    """Base model that forbids unknown keys so config typos fail fast."""

    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    """Dataset location, classes and splitting policy."""

    root: Path
    classes: list[str] = Field(min_length=1)
    image_size: int = Field(gt=0)
    split_strategy: SplitStrategy
    phash_hash_size: int = Field(gt=0)
    phash_distance: int = Field(ge=0)
    test_fraction: float = Field(gt=0.0, lt=1.0)
    val_fraction: float = Field(ge=0.0, lt=1.0)
    exclude_list: Path | None = None

    @field_validator("classes")
    @classmethod
    def _unique_classes(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("`classes` must be unique")
        return value


class ModelConfig(_Strict):
    """Backbone and classifier head."""

    backbone: str
    pretrained: bool
    num_classes: int = Field(gt=1)


class SSLConfig(_Strict):
    """SimCLR self-supervised pretraining hyperparameters."""

    epochs: int = Field(gt=0)
    lr: float = Field(gt=0.0)
    batch_size: int = Field(gt=0)
    weight_decay: float = Field(ge=0.0)
    temperature: float = Field(gt=0.0)
    proj_dim: int = Field(gt=0)
    proj_hidden_dim: int = Field(gt=0)


class FixMatchConfig(_Strict):
    """FixMatch semi-supervised hyperparameters."""

    mu: int = Field(gt=0)
    threshold: float = Field(gt=0.0, le=1.0)
    lambda_u: float = Field(ge=0.0)


class FinetuneConfig(_Strict):
    """Fine-tuning hyperparameters for supervised and FixMatch modes."""

    method: FinetuneMethod
    epochs: int = Field(gt=0)
    lr: float = Field(gt=0.0)
    batch_size: int = Field(gt=0)
    weight_decay: float = Field(ge=0.0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    early_stop_patience: int = Field(default=10, gt=0)
    fixmatch: FixMatchConfig


class ExperimentConfig(_Strict):
    """Experiment grid and runtime settings."""

    label_fractions: list[float] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    device: Device
    output_dir: Path
    workers: int = Field(ge=0)

    @field_validator("label_fractions")
    @classmethod
    def _fractions_in_range(cls, value: list[float]) -> list[float]:
        for fraction in value:
            if not 0.0 < fraction <= 1.0:
                raise ValueError(f"label fraction {fraction} must be in (0, 1]")
        return value


class Config(_Strict):
    """Top-level configuration composed of all per-concern sections."""

    data: DataConfig
    model: ModelConfig
    ssl: SSLConfig
    finetune: FinetuneConfig
    experiment: ExperimentConfig

    @model_validator(mode="after")
    def _check_num_classes(self) -> Config:
        if self.model.num_classes != len(self.data.classes):
            raise ValueError(
                f"model.num_classes ({self.model.num_classes}) must equal "
                f"len(data.classes) ({len(self.data.classes)})"
            )
        return self

    def config_hash(self, length: int = 12) -> str:
        """Return a short, stable content hash of the effective configuration.

        Used as the ``config_hash`` column in ``results.csv`` so every result row
        is traceable to the exact hyperparameters that produced it.

        Args:
            length: Number of leading hex characters of the SHA-256 digest to keep.

        Returns:
            The truncated hex digest of the canonical JSON form of this config.
        """
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` into ``base``, returning a new dict.

    Args:
        base: The base mapping (not mutated).
        overrides: Values that take precedence; nested dicts are merged, not replaced.

    Returns:
        A new merged dictionary.
    """
    merged = dict(base)
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dict, raising :class:`ConfigError` on failure.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed mapping.
    """
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - exercised via malformed files
        raise ConfigError(f"Failed to parse YAML in {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"Expected a mapping at the top of {path}, got {type(loaded).__name__}")
    return loaded


def load_config(
    config_path: str | Path = "configs/config.yaml",
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    """Load, compose, override and validate the project configuration.

    The base file is expected to contain an ``includes`` mapping of
    ``section -> filename``; each referenced file is resolved relative to the base
    file's directory and loaded into that section.

    Args:
        config_path: Path to the base ``config.yaml``.
        overrides: Optional nested mapping deep-merged over the composed config
            before validation (e.g. ``{"finetune": {"method": "supervised"}}``).

    Returns:
        A validated :class:`Config`.

    Raises:
        ConfigError: If a file is missing/malformed or the merged config is invalid.
    """
    base_path = Path(config_path)
    base = _load_yaml(base_path)

    includes = base.get("includes")
    if not isinstance(includes, dict):
        raise ConfigError(f"{base_path} must contain an `includes` mapping of section -> filename")

    composed: dict[str, Any] = {}
    for section, filename in includes.items():
        section_path = base_path.parent / filename
        composed[section] = _load_yaml(section_path)

    if overrides:
        composed = _deep_merge(composed, overrides)

    try:
        return Config.model_validate(composed)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration:\n{exc}") from exc
