"""End-to-end run orchestration.

The glue between config, data, models, training and evaluation.

These functions are deliberately free of any CLI / Typer dependency so they can be
unit-tested directly and reused by the Streamlit app. :mod:`brain_tumor_ssl.cli`
is a thin wrapper that parses arguments and calls into here.

Each entry point resolves the device, seeds every RNG, builds the requested split
and loaders, runs the relevant training loop and (for fine-tuning) records one row
in ``results.csv`` so the paper's tables come straight from the CSV.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from brain_tumor_ssl.config import Config
from brain_tumor_ssl.data.datasets import LabeledSet, TwoViewSet, load_image
from brain_tumor_ssl.data.indexing import Sample, index_dataset
from brain_tumor_ssl.data.splits import DataSplit, make_split, select_labeled
from brain_tumor_ssl.data.transforms import get_transform
from brain_tumor_ssl.evaluation.explain import explain, save_explanation
from brain_tumor_ssl.evaluation.metrics import ClassificationMetrics, evaluate
from brain_tumor_ssl.models.classifier import (
    Classifier,
    build_classifier,
    build_simclr_model,
    load_ssl_into_classifier,
)
from brain_tumor_ssl.training.finetune import finetune_fixmatch, finetune_supervised
from brain_tumor_ssl.training.ssl_simclr import pretrain_simclr
from brain_tumor_ssl.utils.device import resolve_device
from brain_tumor_ssl.utils.io import append_results_row, load_checkpoint, save_checkpoint
from brain_tumor_ssl.utils.logging import get_logger
from brain_tumor_ssl.utils.seed import seed_everything

logger = get_logger()


def _loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    *,
    shuffle: bool,
    drop_last: bool = False,
    device: torch.device,
) -> DataLoader:
    """Build a ``DataLoader`` with project-standard settings.

    Args:
        dataset: The dataset to wrap.
        batch_size: Mini-batch size.
        workers: Number of DataLoader worker processes.
        shuffle: Whether to shuffle each epoch.
        drop_last: Whether to drop a trailing partial batch (needed for the SimCLR
            BatchNorm projection head and for the FixMatch unlabelled stream).
        device: Target device; pinned memory is enabled only on CUDA.

    Returns:
        A configured ``DataLoader``.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=drop_last,
        pin_memory=device.type == "cuda",
    )


def _index_and_split(cfg: Config, seed: int) -> tuple[list[Sample], DataSplit]:
    """Index the dataset and build the configured train/val/test split.

    Args:
        cfg: The validated configuration.
        seed: RNG seed for the split.

    Returns:
        ``(all_samples, split)``.
    """
    samples = index_dataset(cfg.data.root, cfg.data.classes, cfg.data.exclude_list)
    split = make_split(samples, cfg.data, seed)
    return samples, split


def _checkpoints_dir(cfg: Config) -> Path:
    """Return ``<output_dir>/checkpoints`` (not yet created)."""
    return cfg.experiment.output_dir / "checkpoints"


def run_pretrain(cfg: Config, *, checkpoint_path: Path | None = None) -> Path:
    """Run SimCLR self-supervised pretraining over the training images.

    All training-split images are used as unlabelled data (labels are ignored). The
    resulting backbone weights are written to disk for :func:`run_finetune` to load.

    Args:
        cfg: The validated configuration.
        checkpoint_path: Where to write the SSL checkpoint. Defaults to
            ``<output_dir>/checkpoints/simclr.pt``.

    Returns:
        The path of the written checkpoint.
    """
    device = resolve_device(cfg.experiment.device)
    seed_everything(cfg.experiment.seeds[0])
    _, split = _index_and_split(cfg, cfg.experiment.seeds[0])

    view_set = TwoViewSet(split.train, get_transform("simclr", cfg.data.image_size))
    loader = _loader(
        view_set,
        cfg.ssl.batch_size,
        cfg.experiment.workers,
        shuffle=True,
        drop_last=True,
        device=device,
    )

    model = build_simclr_model(cfg.model, cfg.ssl, cfg.data.image_size)
    history = pretrain_simclr(model, loader, cfg.ssl, device)

    path = checkpoint_path or _checkpoints_dir(cfg) / "simclr.pt"
    save_checkpoint(
        {
            "model": model.state_dict(),
            "config_hash": cfg.config_hash(),
            "history": history,
        },
        path,
    )
    return path


def _build_finetune_classifier(
    cfg: Config, ssl_checkpoint: Path | None, device: torch.device
) -> Classifier:
    """Build the classifier for fine-tuning, optionally seeding it with SSL weights.

    Args:
        cfg: The validated configuration.
        ssl_checkpoint: SimCLR checkpoint to transfer into the backbone, or None to
            use the backbone's configured (ImageNet or scratch) initialisation.
        device: Device the classifier is moved to.

    Returns:
        The initialised classifier.
    """
    if ssl_checkpoint is not None:
        # Skip the ImageNet download since the backbone is about to be overwritten.
        clf = build_classifier(
            cfg.model, cfg.data.image_size, pretrained=False, dropout=cfg.finetune.dropout
        )
        state = load_checkpoint(ssl_checkpoint, map_location=device)
        load_ssl_into_classifier(clf, state["model"])
    else:
        clf = build_classifier(cfg.model, cfg.data.image_size, dropout=cfg.finetune.dropout)
    return clf.to(device)


def run_finetune(
    cfg: Config,
    *,
    label_fraction: float,
    seed: int,
    ssl_checkpoint: Path | None = None,
    results_csv: Path | None = None,
    checkpoint_path: Path | None = None,
) -> ClassificationMetrics:
    """Fine-tune a classifier at one ``(label_fraction, seed)`` point and evaluate it.

    Dispatches to supervised or FixMatch training per ``cfg.finetune.method``,
    evaluates the best checkpoint on the held-out test split, and appends one row to
    ``results.csv``.

    Args:
        cfg: The validated configuration.
        label_fraction: Fraction of each class to label (the rest become unlabelled).
        seed: RNG seed for the split, label selection and training.
        ssl_checkpoint: Optional SimCLR checkpoint to initialise the backbone.
        results_csv: Where to append the result row. Defaults to
            ``<output_dir>/results.csv``.
        checkpoint_path: Optional path to save the fine-tuned classifier.

    Returns:
        The test-set :class:`ClassificationMetrics`.
    """
    device = resolve_device(cfg.experiment.device)
    seed_everything(seed)
    _, split = _index_and_split(cfg, seed)
    labeled, unlabeled = select_labeled(split.train, label_fraction, seed)

    size = cfg.data.image_size
    workers = cfg.experiment.workers
    bs = cfg.finetune.batch_size

    labeled_loader = _loader(
        LabeledSet(labeled, get_transform("weak", size)),
        bs,
        workers,
        shuffle=True,
        device=device,
    )
    val_loader = (
        _loader(LabeledSet(split.val, get_transform("eval", size)), bs, workers,
                shuffle=False, device=device)
        if split.val
        else None
    )
    test_loader = _loader(
        LabeledSet(split.test, get_transform("eval", size)), bs, workers,
        shuffle=False, device=device,
    )

    clf = _build_finetune_classifier(cfg, ssl_checkpoint, device)

    if cfg.finetune.method == "fixmatch":
        unlabeled_loader = (
            _loader(
                TwoViewSet(unlabeled, get_transform("weak", size), get_transform("strong", size)),
                bs * cfg.finetune.fixmatch.mu,
                workers,
                shuffle=True,
                device=device,
            )
            if unlabeled
            else None
        )
        finetune_fixmatch(
            clf,
            labeled_loader,
            unlabeled_loader,
            val_loader,
            cfg.finetune,
            device,
            cfg.data.classes,
        )
    else:
        finetune_supervised(
            clf, labeled_loader, val_loader, cfg.finetune, device, cfg.data.classes
        )

    metrics = evaluate(clf, test_loader, device, cfg.data.classes)
    logger.info(
        "[finetune] frac={} seed={} -> acc={:.4f} macroF1={:.4f}",
        label_fraction,
        seed,
        metrics.accuracy,
        metrics.macro_f1,
    )

    append_results_row(
        results_csv or cfg.experiment.output_dir / "results.csv",
        {
            "seed": seed,
            "label_fraction": label_fraction,
            "split": cfg.data.split_strategy,
            "method": cfg.finetune.method,
            "ssl_init": ssl_checkpoint is not None,
            "accuracy": round(metrics.accuracy, 6),
            "macro_f1": round(metrics.macro_f1, 6),
            "config_hash": cfg.config_hash(),
        },
    )

    if checkpoint_path is not None:
        save_checkpoint(
            {"model": clf.state_dict(), "config_hash": cfg.config_hash(), "seed": seed},
            checkpoint_path,
        )
    return metrics


def run_grid(
    cfg: Config,
    *,
    ssl_checkpoint: Path | None = None,
    results_csv: Path | None = None,
) -> list[ClassificationMetrics]:
    """Fine-tune across the full ``seeds`` x ``label_fractions`` experiment grid.

    Args:
        cfg: The validated configuration (its ``experiment`` grid drives the sweep).
        ssl_checkpoint: Optional SimCLR checkpoint to initialise every run.
        results_csv: Where result rows accumulate. Defaults to
            ``<output_dir>/results.csv``.

    Returns:
        One :class:`ClassificationMetrics` per grid point, in iteration order.
    """
    results: list[ClassificationMetrics] = []
    for seed in cfg.experiment.seeds:
        for fraction in cfg.experiment.label_fractions:
            logger.info("[grid] seed={} label_fraction={}", seed, fraction)
            results.append(
                run_finetune(
                    cfg,
                    label_fraction=fraction,
                    seed=seed,
                    ssl_checkpoint=ssl_checkpoint,
                    results_csv=results_csv,
                )
            )
    return results


def run_evaluate(
    cfg: Config,
    *,
    checkpoint_path: Path,
    seed: int | None = None,
    explain_n: int = 0,
    explain_dir: Path | None = None,
) -> ClassificationMetrics:
    """Evaluate a saved classifier checkpoint on the test split.

    Args:
        cfg: The validated configuration (must match the trained model's classes).
        checkpoint_path: Path to a classifier checkpoint written by :func:`run_finetune`.
        seed: Seed used to reproduce the split. Defaults to the checkpoint's recorded
            seed, falling back to the first configured seed.
        explain_n: Number of test images to render attention-rollout overlays for.
        explain_dir: Directory to write overlays into. Defaults to
            ``<output_dir>/explanations``.

    Returns:
        The test-set :class:`ClassificationMetrics`.
    """
    device = resolve_device(cfg.experiment.device)
    state = load_checkpoint(checkpoint_path, map_location=device)
    used_seed = seed if seed is not None else int(state.get("seed", cfg.experiment.seeds[0]))
    seed_everything(used_seed)

    clf = build_classifier(cfg.model, cfg.data.image_size, pretrained=False)
    clf.load_state_dict(state["model"])
    clf.to(device)

    _, split = _index_and_split(cfg, used_seed)
    test_loader = _loader(
        LabeledSet(split.test, get_transform("eval", cfg.data.image_size)),
        cfg.finetune.batch_size,
        cfg.experiment.workers,
        shuffle=False,
        device=device,
    )
    metrics = evaluate(clf, test_loader, device, cfg.data.classes)
    logger.info("[evaluate] acc={:.4f} macroF1={:.4f}", metrics.accuracy, metrics.macro_f1)

    if explain_n > 0 and split.test:
        _save_explanations(cfg, clf, split, device, explain_n, explain_dir)

    return metrics


def _save_explanations(
    cfg: Config,
    clf: Classifier,
    split: DataSplit,
    device: torch.device,
    n: int,
    explain_dir: Path | None,
) -> None:
    """Render attention-rollout overlays for the first ``n`` test images.

    Args:
        cfg: The validated configuration.
        clf: The trained classifier.
        split: The data split whose test images are explained.
        device: Device to run the forward pass on.
        n: Number of images to explain.
        explain_dir: Output directory (defaults to ``<output_dir>/explanations``).
    """
    out_dir = explain_dir or cfg.experiment.output_dir / "explanations"
    transform = get_transform("eval", cfg.data.image_size)
    chosen = split.test[:n]
    images = torch.stack([transform(load_image(s.path)) for s in chosen]).to(device)
    preds, heatmaps = explain(clf, images)
    for i, sample in enumerate(chosen):
        pred_name = cfg.data.classes[int(preds[i])]
        path = out_dir / f"{sample.path.stem}_pred-{pred_name}.png"
        save_explanation(images[i], heatmaps[i], path)
        logger.info("[explain] {} -> {} (pred={})", sample.path.name, path, pred_name)
