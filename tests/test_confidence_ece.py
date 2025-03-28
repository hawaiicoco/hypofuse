"""Expected Calibration Error and reliability bins."""

from __future__ import annotations

import pytest

from hypofuse.confidence import expected_calibration_error, reliability_bins


def test_ece_perfect_calibration_is_zero() -> None:
    confidences = [0.0, 0.5, 1.0]
    accuracies = [0.0, 0.5, 1.0]
    assert expected_calibration_error(confidences, accuracies) == pytest.approx(0.0)


def test_ece_miscalibration_is_positive() -> None:
    confidences = [0.9, 0.9, 0.9]
    accuracies = [0.1, 0.1, 0.1]
    assert expected_calibration_error(confidences, accuracies) > 0


def test_ece_handles_empty_inputs() -> None:
    assert expected_calibration_error([], []) == 0.0


def test_ece_alignment_required() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [0.5, 0.5])


def test_reliability_bins_counts_match() -> None:
    confs = [0.1, 0.2, 0.95, 0.99]
    accs = [0.0, 0.0, 1.0, 1.0]
    bins = reliability_bins(confs, accs, n_bins=4)
    counts = [b.count for b in bins]
    assert sum(counts) == len(confs)


def test_reliability_bins_bin_edges() -> None:
    bins = reliability_bins([0.5], [0.5], n_bins=4)
    assert bins[2].lower == pytest.approx(0.5)
    assert bins[2].upper == pytest.approx(0.75)
