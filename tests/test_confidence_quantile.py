"""Quantile binning for expected calibration error."""

from __future__ import annotations

import pytest

from hypofuse.confidence import expected_calibration_error


def test_quantile_vs_uniform_on_skewed() -> None:
    confs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    accs = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    ece_u = expected_calibration_error(confs, accs, n_bins=5, binning="uniform")
    ece_q = expected_calibration_error(confs, accs, n_bins=5, binning="quantile")
    assert 0.0 <= ece_u <= 1.0
    assert 0.0 <= ece_q <= 1.0
    assert ece_u != pytest.approx(ece_q, abs=1e-6)


def test_quantile_single_bin() -> None:
    # Single bin: avg_conf = mean(0.1, 0.5, 0.9) = 0.5
    # avg_acc = mean(0, 1, 1) = 2/3; ECE = |0.5 - 2/3| = 1/6
    ece = expected_calibration_error([0.1, 0.5, 0.9], [0, 1, 1], n_bins=1, binning="quantile")
    assert ece == pytest.approx(1 / 6)


def test_quantile_default_is_uniform() -> None:
    confs = [0.1, 0.5, 0.9]
    accs = [0, 1, 1]
    ece_default = expected_calibration_error(confs, accs, n_bins=5)
    ece_uniform = expected_calibration_error(confs, accs, n_bins=5, binning="uniform")
    assert ece_default == pytest.approx(ece_uniform)


def test_quantile_unknown_binning() -> None:
    with pytest.raises(ValueError, match=r"unknown binning"):
        expected_calibration_error([0.5], [1], binning="magic")


def test_quantile_handles_empty() -> None:
    assert expected_calibration_error([], [], binning="quantile") == 0.0
