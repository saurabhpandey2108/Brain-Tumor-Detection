"""Dataset splitting: leakage-free perceptual-hash grouped split, naive and source
splits, plus class-balanced labelled/unlabelled subsetting.

The phash strategy approximates patient-level grouping (the public dataset has no
patient IDs): near-duplicate slices are clustered with union-find over perceptual
hashes and whole clusters are kept on one side of every split, so no near-duplicate
can leak across train/val/test.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import imagehash
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from brain_tumor_ssl.config import DataConfig
from brain_tumor_ssl.data.datasets import load_image
from brain_tumor_ssl.data.indexing import Sample, source_partition
from brain_tumor_ssl.utils.logging import get_logger

logger = get_logger()


@dataclass(frozen=True)
class DataSplit:
    """A train/val/test partition of samples.

    Attributes:
        train: Training samples.
        val: Validation samples (may be empty if ``val_fraction == 0``).
        test: Test samples.
    """

    train: list[Sample]
    val: list[Sample]
    test: list[Sample]

    def counts(self) -> dict[str, int]:
        """Return the number of samples in each split partition."""
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def group_samples_by_phash(
    samples: list[Sample], hash_size: int, max_distance: int
) -> list[int]:
    """Cluster near-duplicate samples via union-find over perceptual hashes.

    Two samples join the same group when their perceptual-hash Hamming distance is
    at most ``max_distance``. Comparison is O(n^2); fine for the few-thousand-image
    scale here.

    Args:
        samples: Samples to group (group id is returned per sample, in order).
        hash_size: imagehash ``phash`` size (produces ``hash_size**2`` bits).
        max_distance: Maximum Hamming distance to treat two slices as duplicates.

    Returns:
        A list of contiguous integer group ids, parallel to ``samples``.
    """
    hashes = [imagehash.phash(load_image(s.path), hash_size=hash_size) for s in samples]
    n = len(samples)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if hashes[i] - hashes[j] <= max_distance:
                union(i, j)

    mapping: dict[int, int] = {}
    group_ids: list[int] = []
    for i in range(n):
        root = find(i)
        group_ids.append(mapping.setdefault(root, len(mapping)))
    logger.debug("phash grouping: {} samples -> {} groups", n, len(mapping))
    return group_ids


def _group_holdout(
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out approximately ``fraction`` of samples, group-aware and stratified.

    Args:
        indices: Sample indices to split.
        labels: Per-index class labels (aligned to ``indices``).
        groups: Per-index group ids (aligned to ``indices``).
        fraction: Target fraction to hold out.
        seed: RNG seed for the fold shuffle.

    Returns:
        ``(held_out_indices, remaining_indices)``.

    Raises:
        ValueError: If there are too few groups to honour the split.
    """
    n_groups = len(np.unique(groups))
    n_splits = min(max(2, round(1.0 / fraction)), n_groups)
    if n_splits < 2:
        raise ValueError(
            f"Need at least 2 groups for a group-aware split, found {n_groups}"
        )
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_pos, held_pos = next(iter(skf.split(indices.reshape(-1, 1), labels, groups)))
    return indices[held_pos], indices[rest_pos]


def grouped_split(
    samples: list[Sample],
    group_ids: list[int],
    test_fraction: float,
    val_fraction: float,
    seed: int,
) -> DataSplit:
    """Build a group-aware, stratified train/val/test split.

    No group spans two partitions, so near-duplicates never leak across the split.

    Args:
        samples: All samples.
        group_ids: Group id per sample (parallel to ``samples``).
        test_fraction: Fraction of samples to place in the test split.
        val_fraction: Fraction (of all samples) to place in the validation split.
        seed: RNG seed.

    Returns:
        The resulting :class:`DataSplit`.
    """
    labels = np.array([s.label for s in samples])
    groups = np.array(group_ids)
    idx = np.arange(len(samples))

    test_idx, rest_idx = _group_holdout(idx, labels, groups, test_fraction, seed)
    if val_fraction > 0.0:
        rel = val_fraction / (1.0 - test_fraction)
        val_idx, train_idx = _group_holdout(
            rest_idx, labels[rest_idx], groups[rest_idx], rel, seed
        )
    else:
        val_idx = np.array([], dtype=int)
        train_idx = rest_idx

    def pick(arr: np.ndarray) -> list[Sample]:
        return [samples[i] for i in arr]

    return DataSplit(pick(train_idx), pick(val_idx), pick(test_idx))


def naive_split(
    samples: list[Sample], test_fraction: float, val_fraction: float, seed: int
) -> DataSplit:
    """Build a plain stratified random split (ignores groups).

    This exists ONLY to measure the optimistic leakage inflation versus the
    group-aware phash split; do not report it as the headline result.

    Args:
        samples: All samples.
        test_fraction: Fraction to place in the test split.
        val_fraction: Fraction (of all samples) to place in the validation split.
        seed: RNG seed.

    Returns:
        The resulting :class:`DataSplit`.
    """
    labels = [s.label for s in samples]
    train_val, test = train_test_split(
        samples, test_size=test_fraction, stratify=labels, random_state=seed
    )
    if val_fraction > 0.0:
        rel = val_fraction / (1.0 - test_fraction)
        labels_tv = [s.label for s in train_val]
        train, val = train_test_split(
            train_val, test_size=rel, stratify=labels_tv, random_state=seed
        )
    else:
        train, val = train_val, []
    return DataSplit(train, val, test)


def source_split(samples: list[Sample], val_fraction: float, seed: int) -> DataSplit:
    """Use the dataset's own train/test partition (from ``Tr-``/``Te-`` prefixes).

    A validation set is carved from the training side with a stratified random split.

    Args:
        samples: All samples.
        val_fraction: Fraction of the training side to hold out for validation.
        seed: RNG seed.

    Returns:
        The resulting :class:`DataSplit`.
    """
    train_all = [s for s in samples if source_partition(s) == "train"]
    test = [s for s in samples if source_partition(s) == "test"]
    if not train_all or not test:
        logger.warning(
            "source split is unbalanced (train={}, test={}); the local dataset may "
            "not carry usable Tr-/Te- prefixes",
            len(train_all),
            len(test),
        )

    if val_fraction > 0.0 and len(train_all) > len(set(s.label for s in train_all)):
        labels = [s.label for s in train_all]
        train, val = train_test_split(
            train_all, test_size=val_fraction, stratify=labels, random_state=seed
        )
    else:
        train, val = train_all, []
    return DataSplit(train, val, test)


def make_split(samples: list[Sample], cfg: DataConfig, seed: int) -> DataSplit:
    """Dispatch to the configured split strategy.

    Args:
        samples: All indexed samples.
        cfg: Validated data configuration (selects the strategy and its parameters).
        seed: RNG seed.

    Returns:
        The resulting :class:`DataSplit`.
    """
    if cfg.split_strategy == "phash":
        group_ids = group_samples_by_phash(samples, cfg.phash_hash_size, cfg.phash_distance)
        split = grouped_split(samples, group_ids, cfg.test_fraction, cfg.val_fraction, seed)
    elif cfg.split_strategy == "naive":
        split = naive_split(samples, cfg.test_fraction, cfg.val_fraction, seed)
    else:  # "source"
        split = source_split(samples, cfg.val_fraction, seed)
    logger.info("split '{}' -> {}", cfg.split_strategy, split.counts())
    return split


def select_labeled(
    train: list[Sample], fraction: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    """Class-balanced selection of a labelled subset; the rest become unlabelled.

    At least one sample per class is always labelled, even at very small fractions.

    Args:
        train: Training samples to subset.
        fraction: Fraction of each class to label (``>= 1.0`` labels everything).
        seed: RNG seed for the per-class shuffle.

    Returns:
        ``(labeled, unlabeled)`` lists. When ``fraction >= 1.0`` the unlabelled
        list is empty.
    """
    rng = random.Random(seed)
    by_class: dict[int, list[Sample]] = defaultdict(list)
    for sample in train:
        by_class[sample.label].append(sample)

    labeled: list[Sample] = []
    unlabeled: list[Sample] = []
    for label in sorted(by_class):
        items = by_class[label][:]
        rng.shuffle(items)
        k = len(items) if fraction >= 1.0 else max(1, round(fraction * len(items)))
        labeled.extend(items[:k])
        unlabeled.extend(items[k:])
    logger.info(
        "label selection (fraction={}): {} labelled, {} unlabelled",
        fraction,
        len(labeled),
        len(unlabeled),
    )
    return labeled, unlabeled
