"""SimCLR self-supervised pretraining loop."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from brain_tumor_ssl.config import SSLConfig
from brain_tumor_ssl.models.classifier import SimCLRModel
from brain_tumor_ssl.training.losses import nt_xent
from brain_tumor_ssl.utils.io import save_checkpoint
from brain_tumor_ssl.utils.logging import get_logger

logger = get_logger()


def pretrain_simclr(
    model: SimCLRModel,
    loader: DataLoader,
    cfg: SSLConfig,
    device: torch.device,
    *,
    checkpoint_path: Path | None = None,
    config_hash: str | None = None,
) -> list[float]:
    """Run SimCLR contrastive pretraining.

    Trains for up to ``cfg.epochs``. If ``cfg.early_stop_patience > 0``, training
    stops once the mean epoch NT-Xent loss fails to improve (by more than
    ``cfg.early_stop_min_delta``) for that many consecutive epochs, and the
    lowest-loss weights are restored before returning. With patience ``0`` (the
    default) it runs all epochs exactly as before.

    Args:
        model: The SimCLR model (backbone + projection head).
        loader: DataLoader yielding two augmented views ``(view1, view2)``.
            Should use ``drop_last=True`` so the BatchNorm projection head never
            sees a singleton batch.
        cfg: SimCLR hyperparameters.
        device: Device to train on (AMP is enabled automatically on CUDA).
        checkpoint_path: If given (and ``cfg.checkpoint_every > 0``), a crash-safe
            checkpoint is written here every ``cfg.checkpoint_every`` epochs so a
            long CPU run interrupted near the end is not lost.
        config_hash: Recorded into any periodic checkpoint for traceability.

    Returns:
        Per-epoch mean NT-Xent loss.
    """
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None

    best_loss = float("inf")
    best_state: dict[str, object] | None = None
    num_bad_epochs = 0

    history: list[float] = []
    for epoch in range(cfg.epochs):
        running_loss = 0.0
        num_seen = 0
        model.train()
        for view1, view2 in loader:
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            context = torch.autocast("cuda") if use_amp else nullcontext()
            with context:
                z1 = model(view1)
                z2 = model(view2)
                loss = nt_xent(z1, z2, cfg.temperature)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * view1.size(0)
            num_seen += view1.size(0)

        epoch_loss = running_loss / max(num_seen, 1)
        history.append(epoch_loss)
        logger.info("[SSL] epoch {}/{} loss={:.4f}", epoch + 1, cfg.epochs, epoch_loss)

        if epoch_loss < best_loss - cfg.early_stop_min_delta:
            best_loss = epoch_loss
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            num_bad_epochs = 0
        else:
            num_bad_epochs += 1

        if (
            checkpoint_path is not None
            and cfg.checkpoint_every > 0
            and (epoch + 1) % cfg.checkpoint_every == 0
        ):
            save_checkpoint(
                {"model": model.state_dict(), "config_hash": config_hash, "history": history},
                checkpoint_path,
            )

        if cfg.early_stop_patience > 0 and num_bad_epochs >= cfg.early_stop_patience:
            logger.info(
                "[SSL] early stopping at epoch {} (no loss improvement for {} epochs)",
                epoch + 1,
                cfg.early_stop_patience,
            )
            break

    # Restore the lowest-loss weights so the returned model is the best, not the last.
    if best_state is not None:
        model.load_state_dict(best_state)
    return history
