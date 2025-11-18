"""Synthetic dataset properties and determinism."""

from __future__ import annotations

from hypofuse.neural import FEATURE_NAMES, synthetic_confidence_dataset


def test_exact_length() -> None:
    features, labels = synthetic_confidence_dataset(n=100, seed=0)
    assert len(features) == 100
    assert len(labels) == 100


def test_labels_are_binary() -> None:
    _, labels = synthetic_confidence_dataset(n=200, seed=0)
    assert all(lbl in (0, 1) for lbl in labels)


def test_feature_width() -> None:
    features, _ = synthetic_confidence_dataset(n=50, seed=0)
    for row in features:
        assert len(row) == len(FEATURE_NAMES)


def test_deterministic_for_fixed_seed() -> None:
    x1, y1 = synthetic_confidence_dataset(n=100, seed=42)
    x2, y2 = synthetic_confidence_dataset(n=100, seed=42)
    assert x1 == x2
    assert y1 == y2


def test_different_seeds_differ() -> None:
    x1, _ = synthetic_confidence_dataset(n=100, seed=0)
    x2, _ = synthetic_confidence_dataset(n=100, seed=1)
    assert x1 != x2


def test_correct_group_higher_mean_posterior() -> None:
    # By construction Beta(6,2) mean = 0.75 > Beta(2,6) mean = 0.25
    features, labels = synthetic_confidence_dataset(n=400, seed=0)
    correct_max = [row[0] for row, lbl in zip(features, labels, strict=True) if lbl == 1]
    incorrect_max = [row[0] for row, lbl in zip(features, labels, strict=True) if lbl == 0]
    mean_correct = sum(correct_max) / len(correct_max)
    mean_incorrect = sum(incorrect_max) / len(incorrect_max)
    assert mean_correct > mean_incorrect
