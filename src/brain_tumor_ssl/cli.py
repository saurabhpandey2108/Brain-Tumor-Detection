"""Command-line interface for brain-tumor-ssl.

Thin Typer wrappers around :mod:`brain_tumor_ssl.runner`. Every subcommand loads
and validates the config (with optional overrides), then delegates to a runner
function. The subcommands are:

* ``pretrain`` - SimCLR self-supervised pretraining -> SSL checkpoint.
* ``finetune`` - supervised / FixMatch fine-tuning at one (fraction, seed) point.
* ``evaluate`` - score a saved checkpoint on the test split (optional explanations).
* ``run-grid`` - sweep the full seeds x label_fractions experiment grid.
* ``smoke``    - end-to-end CPU run on a tiny generated synthetic dataset.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

import typer

from brain_tumor_ssl.config import load_config
from brain_tumor_ssl.utils.logging import get_logger

logger = get_logger()

app = typer.Typer(
    name="btssl",
    help="Brain Tumor SSL: SimCLR pretraining + FixMatch finetuning for MRI classification.",
    no_args_is_help=True,
    add_completion=False,
)

_CONFIG_OPT = Annotated[Path, typer.Option("--config", "-c", help="Path to config.yaml.")]


@app.command()
def version() -> None:
    """Print the installed package version."""
    from brain_tumor_ssl import __version__

    typer.echo(__version__)


@app.command()
def pretrain(
    config: _CONFIG_OPT = Path("configs/config.yaml"),
    checkpoint: Annotated[
        Path | None, typer.Option(help="Output SSL checkpoint path.")
    ] = None,
) -> None:
    """Run SimCLR self-supervised pretraining on the unlabelled training images."""
    from brain_tumor_ssl.runner import run_pretrain

    cfg = load_config(config)
    path = run_pretrain(cfg, checkpoint_path=checkpoint)
    typer.echo(f"SSL checkpoint written to {path}")


@app.command()
def finetune(
    config: _CONFIG_OPT = Path("configs/config.yaml"),
    label_fraction: Annotated[
        float, typer.Option("--label-fraction", "-f", help="Labelled fraction in (0, 1].")
    ] = 1.0,
    seed: Annotated[int, typer.Option(help="RNG seed.")] = 42,
    ssl_checkpoint: Annotated[
        Path | None, typer.Option("--ssl-checkpoint", help="SimCLR checkpoint to transfer.")
    ] = None,
    method: Annotated[
        str | None,
        typer.Option(help="Override finetune.method: 'supervised' or 'fixmatch'."),
    ] = None,
    output: Annotated[
        Path | None, typer.Option(help="Path to save the fine-tuned classifier.")
    ] = None,
) -> None:
    """Fine-tune a classifier at one (label_fraction, seed) point and evaluate it."""
    from brain_tumor_ssl.runner import run_finetune

    overrides = {"finetune": {"method": method}} if method is not None else None
    cfg = load_config(config, overrides=overrides)
    metrics = run_finetune(
        cfg,
        label_fraction=label_fraction,
        seed=seed,
        ssl_checkpoint=ssl_checkpoint,
        checkpoint_path=output,
    )
    typer.echo(f"accuracy={metrics.accuracy:.4f} macro_f1={metrics.macro_f1:.4f}")


@app.command()
def evaluate(
    checkpoint: Annotated[Path, typer.Argument(help="Classifier checkpoint to evaluate.")],
    config: _CONFIG_OPT = Path("configs/config.yaml"),
    seed: Annotated[
        int | None, typer.Option(help="Seed to reproduce the split (default: checkpoint's).")
    ] = None,
    explain: Annotated[
        int, typer.Option(help="Number of test images to render attention overlays for.")
    ] = 0,
) -> None:
    """Evaluate a saved classifier on the test split and print metrics."""
    from brain_tumor_ssl.runner import run_evaluate

    cfg = load_config(config)
    metrics = run_evaluate(cfg, checkpoint_path=checkpoint, seed=seed, explain_n=explain)
    typer.echo(metrics.report)
    typer.echo(f"accuracy={metrics.accuracy:.4f} macro_f1={metrics.macro_f1:.4f}")


@app.command(name="run-grid")
def run_grid(
    config: _CONFIG_OPT = Path("configs/config.yaml"),
    ssl_checkpoint: Annotated[
        Path | None, typer.Option("--ssl-checkpoint", help="SimCLR checkpoint to transfer.")
    ] = None,
) -> None:
    """Sweep the full seeds x label_fractions grid, accumulating results.csv."""
    from brain_tumor_ssl.runner import run_grid as _run_grid

    cfg = load_config(config)
    results = _run_grid(cfg, ssl_checkpoint=ssl_checkpoint)
    typer.echo(f"completed {len(results)} runs -> {cfg.experiment.output_dir / 'results.csv'}")


@app.command()
def smoke(
    output: Annotated[
        Path | None, typer.Option(help="Output dir (default: a temp directory).")
    ] = None,
) -> None:
    """Run a tiny end-to-end pipeline on synthetic data to verify wiring (CPU-only).

    Generates a small synthetic dataset, then runs SimCLR pretraining, FixMatch
    fine-tuning and test evaluation with a tiny ViT so the whole path executes in
    seconds without a GPU or the real dataset.
    """
    from brain_tumor_ssl.data.synthetic import generate_synthetic_dataset
    from brain_tumor_ssl.runner import run_evaluate, run_finetune, run_pretrain

    classes = ["glioma", "meningioma", "notumor", "pituitary"]
    work = output or Path(tempfile.mkdtemp(prefix="btssl-smoke-"))
    data_root = work / "data"
    generate_synthetic_dataset(data_root, classes, per_class=10, image_size=32, seed=0)

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
            "output_dir": str(work / "results"),
        },
    }
    cfg = load_config("configs/config.yaml", overrides=overrides)

    ckpt = run_pretrain(cfg)
    clf_ckpt = work / "results" / "checkpoints" / "smoke_clf.pt"
    run_finetune(cfg, label_fraction=0.5, seed=42, ssl_checkpoint=ckpt, checkpoint_path=clf_ckpt)
    metrics = run_evaluate(cfg, checkpoint_path=clf_ckpt, seed=42, explain_n=1)

    typer.echo(
        f"smoke OK: acc={metrics.accuracy:.4f} macro_f1={metrics.macro_f1:.4f}\n"
        f"artifacts under {work}"
    )


if __name__ == "__main__":
    app()
