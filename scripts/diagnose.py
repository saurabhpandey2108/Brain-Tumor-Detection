"""Low-level correctness diagnostics for the brain-tumor-ssl pipeline (CPU-only).

Unlike the wiring tests, this verifies the *learning machinery* actually works
before committing GPU/HPC time:

1. NT-Xent loss matches an independent reference implementation (numerics).
2. SimCLR pretraining drives the contrastive loss down (gradients + optimiser).
3. A classifier can learn a *separable* synthetic dataset (forward/backward/label
   mapping all correct) — pure-noise data can't test this.
4. Fine-tuning is deterministic for a fixed seed.
5. FixMatch reaches good accuracy on learnable data and its pseudo-label path runs.

Run: ``uv run python scripts/diagnose.py``
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from brain_tumor_ssl.config import Config, load_config
from brain_tumor_ssl.data.datasets import LabeledSet, TwoViewSet
from brain_tumor_ssl.data.indexing import index_dataset
from brain_tumor_ssl.data.splits import make_split
from brain_tumor_ssl.data.transforms import get_transform
from brain_tumor_ssl.evaluation.metrics import evaluate
from brain_tumor_ssl.models.classifier import build_classifier, build_simclr_model
from brain_tumor_ssl.runner import run_finetune
from brain_tumor_ssl.training.finetune import finetune_supervised
from brain_tumor_ssl.training.losses import nt_xent
from brain_tumor_ssl.training.ssl_simclr import pretrain_simclr
from brain_tumor_ssl.utils.device import resolve_device
from brain_tumor_ssl.utils.seed import seed_everything

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    """Record and print one check's outcome."""
    status = PASS if ok else FAIL
    results.append((status, name, detail))
    print(f"[{status}] {name}: {detail}")


#: Per-class (top-half, bottom-half) brightness. Distinguishing the four classes
#: needs *spatial* reading (classes 1 and 2 share a mean), and the encoding is
#: invariant to horizontal flip and small vertical translation, so it survives the
#: weak augmentation. This is deliberately easy: the goal is to test the training
#: machinery, not its capacity.
_CLASS_PROFILE: dict[str, tuple[float, float]] = {
    "glioma": (0.2, 0.2),
    "meningioma": (0.85, 0.2),
    "notumor": (0.2, 0.85),
    "pituitary": (0.85, 0.85),
}


def make_learnable_dataset(root: Path, per_class: int = 24, size: int = 32, seed: int = 0) -> None:
    """Write an easy, class-separable synthetic dataset to disk.

    Each class has a fixed top-half/bottom-half brightness profile plus per-image
    Gaussian noise (so images and their perceptual hashes differ). The signal is
    flip- and translate-invariant, so a correct classifier must reach near-perfect
    *training* accuracy on it.
    """
    rng = np.random.default_rng(seed)
    mid = size // 2
    for class_name in CLASSES:
        class_dir = root / class_name / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        top, bottom = _CLASS_PROFILE[class_name]
        n_train = round(0.7 * per_class)
        for i in range(per_class):
            img = np.empty((size, size), dtype=np.float64)
            img[:mid, :] = top
            img[mid:, :] = bottom
            img = np.clip(img + rng.normal(0.0, 0.12, (size, size)), 0.0, 1.0)
            arr = (img * 255).astype(np.uint8)
            prefix = "Tr" if i < n_train else "Te"
            name = f"{prefix}-{class_name[:2]}_{i:03d}.png"
            Image.fromarray(arr, mode="L").save(class_dir / name)


def learnable_cfg(data_root: Path, out_dir: Path, *, method: str, epochs: int) -> Config:
    """Build a tiny-ViT CPU config pointed at the learnable dataset."""
    return load_config(
        "configs/config.yaml",
        overrides={
            "data": {"root": str(data_root), "image_size": 32, "split_strategy": "naive"},
            "model": {"backbone": "vit_tiny_patch16_224", "pretrained": False},
            "finetune": {
                "method": method,
                "epochs": epochs,
                "batch_size": 8,
                "lr": 1.0e-3,
                "early_stop_patience": epochs,
            },
            "experiment": {
                "device": "cpu",
                "workers": 0,
                "label_fractions": [1.0],
                "seeds": [42],
                "output_dir": str(out_dir),
            },
        },
    )


def check_nt_xent_numerics() -> None:
    """NT-Xent must match an independent reference within floating-point tolerance."""
    seed_everything(0)
    z1 = torch.randn(5, 16)
    z2 = torch.randn(5, 16)
    temperature = 0.3
    got = float(nt_xent(z1, z2, temperature))

    # Independent reference: full 2B softmax cross-entropy with self entries removed.
    feats = torch.nn.functional.normalize(torch.cat([z1, z2]), dim=1)
    sim = feats @ feats.t() / temperature
    losses = []
    n = 5
    for i in range(2 * n):
        positive = i + n if i < n else i - n
        row = sim[i].clone()
        row[i] = float("-inf")
        log_denom = torch.logsumexp(row, dim=0)
        losses.append((log_denom - row[positive]).item())
    ref = float(np.mean(losses))
    record("nt_xent numerics", abs(got - ref) < 1e-5, f"impl={got:.6f} ref={ref:.6f}")


def check_simclr_loss_decreases(data_root: Path) -> None:
    """SimCLR contrastive loss should fall over a handful of epochs."""
    device = resolve_device("cpu")
    seed_everything(0)
    cfg = learnable_cfg(data_root, Path(tempfile.mkdtemp()), method="fixmatch", epochs=1)
    samples = index_dataset(cfg.data.root, cfg.data.classes)
    split = make_split(samples, cfg.data, 0)
    view_set = TwoViewSet(split.train, get_transform("simclr", cfg.data.image_size))
    loader = torch.utils.data.DataLoader(view_set, batch_size=8, shuffle=True, drop_last=True)

    cfg_ssl = cfg.ssl.model_copy(update={"epochs": 8, "lr": 1.0e-3})
    model = build_simclr_model(cfg.model, cfg_ssl, cfg.data.image_size)
    history = pretrain_simclr(model, loader, cfg_ssl, device)
    drop = history[0] - history[-1]
    record(
        "simclr loss decreases",
        history[-1] < history[0] and drop > 0.05,
        f"first={history[0]:.3f} last={history[-1]:.3f} (drop={drop:.3f})",
    )


def check_classifier_overfits(data_root: Path) -> None:
    """The canonical sanity check: gradients must drive *train* accuracy to ~1.0.

    Augmentation is turned off (eval transform on the training images) so the model
    can memorise. If this fails, the forward/backward/optimizer/label-mapping path is
    broken; if it passes, any weak *test* accuracy is a capacity/data issue, not a bug.
    """
    device = resolve_device("cpu")
    seed_everything(0)
    epochs = 120
    cfg = learnable_cfg(data_root, Path(tempfile.mkdtemp()), method="supervised", epochs=epochs)
    samples = index_dataset(cfg.data.root, cfg.data.classes)
    split = make_split(samples, cfg.data, 0)
    transform = get_transform("eval", cfg.data.image_size)
    loader = torch.utils.data.DataLoader(
        LabeledSet(split.train, transform), batch_size=8, shuffle=True
    )

    clf = build_classifier(cfg.model, cfg.data.image_size, pretrained=False).to(device)
    finetune_supervised(clf, loader, None, cfg.finetune, device, cfg.data.classes)
    m = evaluate(clf, loader, device, cfg.data.classes)
    record(
        "classifier overfits train set",
        m.accuracy > 0.90,
        f"train acc={m.accuracy:.3f} after {epochs} epochs; confusion={m.confusion_matrix}",
    )


def check_classifier_generalizes(data_root: Path) -> None:
    """With augmentation on, a held-out test set should still beat chance comfortably."""
    cfg = learnable_cfg(data_root, Path(tempfile.mkdtemp()), method="supervised", epochs=25)
    metrics = run_finetune(cfg, label_fraction=1.0, seed=42)
    record(
        "supervised generalizes to test",
        metrics.accuracy > 0.60,
        f"test acc={metrics.accuracy:.3f} macroF1={metrics.macro_f1:.3f} (chance=0.25)",
    )


def check_determinism(data_root: Path) -> None:
    """Two fine-tune runs with the same seed must give the same test accuracy."""
    acc = []
    for _ in range(2):
        cfg = learnable_cfg(data_root, Path(tempfile.mkdtemp()), method="supervised", epochs=4)
        acc.append(run_finetune(cfg, label_fraction=1.0, seed=123).accuracy)
    record(
        "determinism (fixed seed)",
        abs(acc[0] - acc[1]) < 1e-6,
        f"run1={acc[0]:.6f} run2={acc[1]:.6f}",
    )


def check_fixmatch_learns(data_root: Path) -> None:
    """FixMatch at a low label fraction should still beat chance on learnable data."""
    cfg = learnable_cfg(data_root, Path(tempfile.mkdtemp()), method="fixmatch", epochs=15)
    metrics = run_finetune(cfg, label_fraction=0.5, seed=42)
    record(
        "fixmatch learns (50% labels)",
        metrics.accuracy > 0.50,
        f"test acc={metrics.accuracy:.3f} macroF1={metrics.macro_f1:.3f}",
    )


def main() -> int:
    """Run all diagnostics and return a process exit code (0 = all passed)."""
    work = Path(tempfile.mkdtemp(prefix="btssl-diag-"))
    data_root = work / "data"
    make_learnable_dataset(data_root)
    print(f"learnable dataset -> {data_root}\n")

    check_nt_xent_numerics()
    check_simclr_loss_decreases(data_root)
    check_classifier_overfits(data_root)
    check_classifier_generalizes(data_root)
    check_determinism(data_root)
    check_fixmatch_learns(data_root)

    failed = [r for r in results if r[0] == FAIL]
    print("\n" + "=" * 60)
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    for status, name, detail in results:
        print(f"  [{status}] {name} — {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
