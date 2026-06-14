"""Command-line interface for brain-tumor-ssl.

This is a placeholder skeleton wired up in Stage 1 so the ``btssl`` entry point
resolves. The real subcommands (pretrain, finetune, evaluate, run-grid, smoke)
are implemented in Stage 7.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="btssl",
    help="Brain Tumor SSL: SimCLR pretraining + FixMatch finetuning for MRI classification.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print the installed package version."""
    from brain_tumor_ssl import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
