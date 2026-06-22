"""Generate a tiny synthetic dataset for CPU smoke tests.

Images are smooth low-frequency random patterns (distinct perceptual hashes), with
a few deliberate near-duplicate pairs per class so the perceptual-hash grouped
split can be tested. Filenames carry ``Tr-``/``Te-`` prefixes so the source split
is testable too. The on-disk layout mirrors the real dataset's nesting:
``root/<class>/<class>/*.png``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class SyntheticManifest:
    """Description of a generated synthetic dataset.

    Attributes:
        root: Dataset root directory.
        samples: ``(path, class_name)`` for every generated image.
        duplicate_pairs: ``(original, near_duplicate)`` path pairs that should be
            grouped together by the perceptual-hash split.
    """

    root: Path
    samples: list[tuple[Path, str]] = field(default_factory=list)
    duplicate_pairs: list[tuple[Path, Path]] = field(default_factory=list)


def _render(seed_array: np.ndarray, image_size: int) -> Image.Image:
    """Upscale a small seed array into a smooth grayscale image.

    Args:
        seed_array: Low-resolution ``uint8`` seed (low spatial frequency).
        image_size: Output square side length.

    Returns:
        A PIL grayscale image.
    """
    img = Image.fromarray(seed_array, mode="L")
    return img.resize((image_size, image_size), Image.BICUBIC)


def generate_synthetic_dataset(
    root: Path,
    classes: list[str],
    per_class: int = 12,
    image_size: int = 64,
    seed: int = 0,
    near_dup_pairs_per_class: int = 2,
    dup_noise: float = 2.0,
    train_ratio: float = 0.7,
) -> SyntheticManifest:
    """Write a small random dataset to disk and return its manifest.

    Args:
        root: Directory to create the dataset under.
        classes: Class names; one sub-tree is created per class.
        per_class: Number of base (non-duplicate) images per class.
        image_size: Side length of each saved image.
        seed: RNG seed for reproducible generation.
        near_dup_pairs_per_class: Number of near-duplicate images added per class.
        dup_noise: Std-dev of the noise added to create a near-duplicate.
        train_ratio: Fraction of base images given the ``Tr-`` prefix (rest ``Te-``).

    Returns:
        A :class:`SyntheticManifest` describing every file written.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    manifest = SyntheticManifest(root=root)

    for class_name in classes:
        class_dir = root / class_name / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        abbr = class_name[:2]
        n_train = round(train_ratio * per_class)

        base_seeds: list[np.ndarray] = []
        base_paths: list[Path] = []
        base_prefixes: list[str] = []
        for i in range(per_class):
            seed_array = rng.integers(0, 256, size=(8, 8), dtype=np.uint8)
            prefix = "Tr" if i < n_train else "Te"
            path = class_dir / f"{prefix}-{abbr}_{i:03d}.png"
            _render(seed_array, image_size).save(path)
            base_seeds.append(seed_array)
            base_paths.append(path)
            base_prefixes.append(prefix)
            manifest.samples.append((path, class_name))

        for d in range(min(near_dup_pairs_per_class, per_class)):
            noisy = base_seeds[d].astype(np.float64) + rng.normal(0.0, dup_noise, size=(8, 8))
            seed_array = np.clip(noisy, 0, 255).astype(np.uint8)
            prefix = base_prefixes[d]
            dup_path = class_dir / f"{prefix}-{abbr}_{d:03d}_dup.png"
            _render(seed_array, image_size).save(dup_path)
            manifest.samples.append((dup_path, class_name))
            manifest.duplicate_pairs.append((base_paths[d], dup_path))

    return manifest
