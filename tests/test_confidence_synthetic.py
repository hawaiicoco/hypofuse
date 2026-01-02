"""Synthetic calibration benchmark set."""

from __future__ import annotations

from hypofuse.confidence import (
    expected_calibration_error,
    synthetic_calibration_set,
)


def test_synthetic_deterministic() -> None:
    s1, l1, p1 = synthetic_calibration_set(n=100, seed=42)
    s2, l2, p2 = synthetic_calibration_set(n=100, seed=42)
    assert s1 == s2
    assert l1 == l2
    assert p1 == p2


def test_synthetic_lengths() -> None:
    scores, labels, p_true = synthetic_calibration_set(n=200, seed=0)
    assert len(scores) == 200
    assert len(labels) == 200
    assert len(p_true) == 200


def test_synthetic_labels_binary() -> None:
    _, labels, _ = synthetic_calibration_set(n=500, seed=0)
    assert all(y in (0, 1) for y in labels)


def test_synthetic_sharpness_one_low_ece() -> None:
    scores, labels, _ = synthetic_calibration_set(n=2000, seed=0, sharpness=1.0)
    float_labels = [float(y) for y in labels]
    ece = expected_calibration_error(scores, float_labels, n_bins=10)
    assert ece < 0.1


def test_synthetic_distorted_higher_ece() -> None:
    scores, labels, _ = synthetic_calibration_set(n=2000, seed=0, sharpness=1.0)
    float_labels = [float(y) for y in labels]
    distorted = [s**3 for s in scores]
    ece_raw = expected_calibration_error(scores, float_labels, n_bins=10)
    ece_dist = expected_calibration_error(distorted, float_labels, n_bins=10)
    assert ece_dist > ece_raw


def test_synthetic_p_true_in_unit_interval() -> None:
    _, _, p_true = synthetic_calibration_set(n=100, seed=0, sharpness=2.0)
    assert all(0.0 <= p <= 1.0 for p in p_true)
