"""Tests for indexing, perceptual-hash grouping, splitting and label selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain_tumor_ssl.data.indexing import index_dataset
from brain_tumor_ssl.data.splits import (
    group_samples_by_phash,
    grouped_split,
    naive_split,
    select_labeled,
)
from brain_tumor_ssl.data.synthetic import generate_synthetic_dataset

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
HASH_SIZE = 8
MAX_DISTANCE = 5


@pytest.fixture
def dataset(tmp_path: Path):
    manifest = generate_synthetic_dataset(
        tmp_path / "synth", CLASSES, per_class=12, image_size=32, seed=7
    )
    samples = index_dataset(tmp_path / "synth", CLASSES)
    return manifest, samples


def test_indexing_finds_all_images(dataset) -> None:
    manifest, samples = dataset
    assert len(samples) == len(manifest.samples)
    assert {s.class_name for s in samples} == set(CLASSES)
    # Labels are the position of the class name in CLASSES.
    for sample in samples:
        assert CLASSES[sample.label] == sample.class_name


def test_near_duplicates_land_in_same_group(dataset) -> None:
    manifest, samples = dataset
    group_ids = group_samples_by_phash(samples, HASH_SIZE, MAX_DISTANCE)
    group_of = {s.path: gid for s, gid in zip(samples, group_ids, strict=True)}
    for original, duplicate in manifest.duplicate_pairs:
        assert group_of[original] == group_of[duplicate], (
            f"near-duplicate {duplicate.name} not grouped with {original.name}"
        )


def test_grouped_split_keeps_groups_intact_and_disjoint(dataset) -> None:
    _, samples = dataset
    group_ids = group_samples_by_phash(samples, HASH_SIZE, MAX_DISTANCE)
    group_of = {s.path: gid for s, gid in zip(samples, group_ids, strict=True)}

    split = grouped_split(samples, group_ids, test_fraction=0.25, val_fraction=0.15, seed=42)

    # Every sample is placed exactly once.
    all_paths = [s.path for part in (split.train, split.val, split.test) for s in part]
    assert len(all_paths) == len(samples)
    assert len(set(all_paths)) == len(samples)

    # No group id appears in more than one partition (no leakage across the split).
    groups_per_part = [
        {group_of[s.path] for s in part} for part in (split.train, split.val, split.test)
    ]
    train_g, val_g, test_g = groups_per_part
    assert train_g.isdisjoint(test_g)
    assert train_g.isdisjoint(val_g)
    assert val_g.isdisjoint(test_g)


def test_grouped_split_has_non_empty_partitions(dataset) -> None:
    _, samples = dataset
    group_ids = group_samples_by_phash(samples, HASH_SIZE, MAX_DISTANCE)
    split = grouped_split(samples, group_ids, test_fraction=0.25, val_fraction=0.15, seed=42)
    assert split.train and split.val and split.test


def test_naive_split_is_stratified(dataset) -> None:
    _, samples = dataset
    split = naive_split(samples, test_fraction=0.25, val_fraction=0.0, seed=42)
    assert len(split.val) == 0
    assert len(split.train) + len(split.test) == len(samples)
    # Every class is represented in the test split.
    assert {s.label for s in split.test} == set(range(len(CLASSES)))


def test_select_labeled_is_class_balanced_and_partitions_train(dataset) -> None:
    _, samples = dataset
    labeled, unlabeled = select_labeled(samples, fraction=0.5, seed=42)

    # Labelled + unlabelled exactly reconstructs the input, disjoint.
    assert len(labeled) + len(unlabeled) == len(samples)
    assert {s.path for s in labeled}.isdisjoint({s.path for s in unlabeled})
    # Each class contributes labelled samples.
    assert {s.label for s in labeled} == set(range(len(CLASSES)))


def test_select_labeled_full_fraction_has_no_unlabeled(dataset) -> None:
    _, samples = dataset
    labeled, unlabeled = select_labeled(samples, fraction=1.0, seed=0)
    assert len(unlabeled) == 0
    assert len(labeled) == len(samples)


def test_select_labeled_tiny_fraction_keeps_one_per_class(dataset) -> None:
    _, samples = dataset
    labeled, _ = select_labeled(samples, fraction=0.01, seed=0)
    counts = dict.fromkeys(range(len(CLASSES)), 0)
    for sample in labeled:
        counts[sample.label] += 1
    assert all(c >= 1 for c in counts.values())
