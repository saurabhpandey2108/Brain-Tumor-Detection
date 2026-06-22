"""Fine-tuning loops: supervised baseline and FixMatch semi-supervised training."""

from __future__ import annotations

import copy
from contextlib import AbstractContextManager, nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from brain_tumor_ssl.config import FinetuneConfig
from brain_tumor_ssl.evaluation.metrics import evaluate
from brain_tumor_ssl.training.callbacks import EarlyStopper
from brain_tumor_ssl.utils.logging import get_logger

logger = get_logger()


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency ("balanced") class weights for cross-entropy.

    Rarer classes receive proportionally larger weights so an imbalanced training
    set does not bias the classifier toward majority classes. Uses the sklearn
    "balanced" formula ``n_samples / (n_classes * count[c])``; absent classes are
    clamped to a count of 1 to avoid division by zero.

    Args:
        labels: Integer class labels of the labelled training samples.
        num_classes: Total number of classes.

    Returns:
        A ``(num_classes,)`` float tensor of per-class weights.
    """
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes).float()
    return counts.sum() / (num_classes * counts.clamp_min(1.0))


def _amp_context(use_amp: bool) -> AbstractContextManager:
    """Return an autocast context on CUDA, else a no-op context."""
    return torch.autocast("cuda") if use_amp else nullcontext()


def _backward(loss: torch.Tensor, optimizer: torch.optim.Optimizer, scaler: object) -> None:
    """Run a backward + optimizer step, with or without an AMP grad scaler."""
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()


def _finalise(model: nn.Module, stopper: EarlyStopper) -> None:
    """Restore the best (or, if none recorded, current) weights into the model."""
    if stopper.best_state is None:
        stopper.best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    model.load_state_dict(stopper.best_state)


def _epoch_end(
    model: nn.Module,
    val_loader: DataLoader | None,
    device: torch.device,
    class_names: list[str],
    epoch: int,
    stopper: EarlyStopper,
) -> bool:
    """Evaluate on validation, update the early stopper, and report stop signal.

    Args:
        model: The model being trained.
        val_loader: Validation loader (if None, the current epoch is snapshotted).
        device: Device to evaluate on.
        class_names: Class names for metric computation.
        epoch: Current epoch index.
        stopper: The early stopper to update.

    Returns:
        True if training should stop early.
    """
    if val_loader is None:
        stopper.best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        return False
    metrics = evaluate(model, val_loader, device, class_names)
    stopper.update(metrics.macro_f1, epoch, model)
    logger.info(
        "epoch {}: val acc={:.4f} macroF1={:.4f}", epoch, metrics.accuracy, metrics.macro_f1
    )
    return stopper.should_stop


def _next_unlabeled(
    unl_iter: object, unlabeled_loader: DataLoader
) -> tuple[object, tuple[torch.Tensor, torch.Tensor]] | None:
    """Fetch the next unlabelled ``(weak, strong)`` batch, restarting on exhaustion.

    Args:
        unl_iter: The current iterator over ``unlabeled_loader``.
        unlabeled_loader: The loader to restart from when the iterator is exhausted.

    Returns:
        ``(iterator, (weak, strong))`` with a possibly-renewed iterator, or ``None``
        if the loader yields no batches at all (so the caller falls back to a purely
        supervised step).
    """
    try:
        return unl_iter, next(unl_iter)
    except StopIteration:
        unl_iter = iter(unlabeled_loader)
        try:
            return unl_iter, next(unl_iter)
        except StopIteration:
            return None


def finetune_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: FinetuneConfig,
    device: torch.device,
    class_names: list[str],
    class_weights: torch.Tensor | None = None,
) -> EarlyStopper:
    """Supervised fine-tuning on the labelled subset (baseline).

    Args:
        model: Classifier returning logits.
        train_loader: Labelled DataLoader yielding ``(images, labels)``.
        val_loader: Validation loader for best-checkpoint selection (may be None).
        cfg: Fine-tuning hyperparameters.
        device: Device to train on.
        class_names: Class names for validation metrics.
        class_weights: Optional per-class cross-entropy weights for imbalance
            (see :func:`compute_class_weights`); ``None`` for unweighted CE.

    Returns:
        The :class:`EarlyStopper` holding the best weights/metric.
    """
    model.to(device)
    weight = class_weights.to(device) if class_weights is not None else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None
    stopper = EarlyStopper(cfg.early_stop_patience)

    for epoch in range(cfg.epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(use_amp):
                loss = F.cross_entropy(model(images), labels, weight=weight)
            _backward(loss, optimizer, scaler)

        if _epoch_end(model, val_loader, device, class_names, epoch, stopper):
            logger.info("early stopping at epoch {}", epoch)
            break

    _finalise(model, stopper)
    return stopper


def finetune_fixmatch(
    model: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader | None,
    val_loader: DataLoader | None,
    cfg: FinetuneConfig,
    device: torch.device,
    class_names: list[str],
    class_weights: torch.Tensor | None = None,
) -> EarlyStopper:
    """FixMatch semi-supervised fine-tuning.

    Each step combines supervised cross-entropy on the labelled batch with a
    consistency loss: confident pseudo-labels from the weakly-augmented unlabelled
    view supervise the model's prediction on the strongly-augmented view. When no
    unlabelled data is available (e.g. 100% label fraction) this reduces to the
    supervised loss.

    Args:
        model: Classifier returning logits.
        labeled_loader: Labelled loader yielding ``(images, labels)``.
        unlabeled_loader: Two-view loader yielding ``(weak, strong)`` (may be None).
        val_loader: Validation loader for best-checkpoint selection (may be None).
        cfg: Fine-tuning hyperparameters (uses ``cfg.fixmatch``).
        device: Device to train on.
        class_names: Class names for validation metrics.
        class_weights: Optional per-class weights for the labelled cross-entropy
            term (the unlabelled consistency loss is left unweighted, as its targets
            are model pseudo-labels); ``None`` for unweighted CE.

    Returns:
        The :class:`EarlyStopper` holding the best weights/metric.
    """
    fm = cfg.fixmatch
    model.to(device)
    weight = class_weights.to(device) if class_weights is not None else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None
    stopper = EarlyStopper(cfg.early_stop_patience)

    for epoch in range(cfg.epochs):
        model.train()
        unl_iter = iter(unlabeled_loader) if unlabeled_loader is not None else None

        for images_l, labels_l in labeled_loader:
            images_l = images_l.to(device, non_blocking=True)
            labels_l = labels_l.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with _amp_context(use_amp):
                loss_sup = F.cross_entropy(model(images_l), labels_l, weight=weight)
                loss_u = torch.zeros((), device=device)

                if unl_iter is not None:
                    batch = _next_unlabeled(unl_iter, unlabeled_loader)
                    if batch is None:
                        # The unlabelled loader yields no batches; act supervised-only.
                        unl_iter = None
                    else:
                        unl_iter, (weak, strong) = batch
                if unl_iter is not None:
                    weak = weak.to(device, non_blocking=True)
                    strong = strong.to(device, non_blocking=True)

                    with torch.no_grad():
                        probs = F.softmax(model(weak), dim=1)
                        max_probs, pseudo = probs.max(dim=1)
                        mask = (max_probs >= fm.threshold).float()

                    if mask.sum() > 0:
                        ce_u = F.cross_entropy(model(strong), pseudo, reduction="none")
                        loss_u = (ce_u * mask).mean()

                loss = loss_sup + fm.lambda_u * loss_u

            _backward(loss, optimizer, scaler)

        if _epoch_end(model, val_loader, device, class_names, epoch, stopper):
            logger.info("early stopping at epoch {}", epoch)
            break

    _finalise(model, stopper)
    return stopper
