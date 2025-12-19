"""Logistic calibrator via gradient descent."""

from __future__ import annotations

import math
import random

import pytest

from hypofuse.confidence import (
    LogisticCalibrator,
    calibrate,
    expected_calibration_error,
    fit_logistic,
)
from hypofuse.exceptions import CalibrationError


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def test_fit_logistic_reduces_ece() -> None:
    rng = random.Random(42)
    scores = [rng.uniform(-2, 2) for _ in range(500)]
    labels = [1.0 if rng.random() < _sigmoid(2 * s - 1) else 0.0 for s in scores]
    raw_probs = [_sigmoid(s) for s in scores]
    raw_ece = expected_calibration_error(raw_probs, labels, n_bins=10)
    cal = fit_logistic(scores, labels)
    cal_probs = cal.apply(scores)
    cal_ece = expected_calibration_error(cal_probs, labels, n_bins=10)
    assert cal_ece < raw_ece


def test_fit_logistic_deterministic() -> None:
    scores = [0.1, 0.5, 0.9, 0.3, 0.7]
    labels = [0.0, 1.0, 1.0, 0.0, 1.0]
    cal1 = fit_logistic(scores, labels)
    cal2 = fit_logistic(scores, labels)
    assert cal1.coef == cal2.coef
    assert cal1.intercept == cal2.intercept


def test_logistic_apply_stays_in_unit_interval() -> None:
    cal = LogisticCalibrator(coef=2.0, intercept=-1.0)
    out = cal.apply([-10.0, 0.0, 10.0])
    for v in out:
        assert 0.0 < v < 1.0


def test_fit_logistic_degenerate_raises() -> None:
    with pytest.raises(CalibrationError, match=r"same class"):
        fit_logistic([0.1, 0.5, 0.9], [1, 1, 1])


def test_calibrate_logistic_method() -> None:
    scores = [0.1, 0.5, 0.9, 0.3, 0.7]
    labels = [0.0, 1.0, 1.0, 0.0, 1.0]
    out = calibrate(scores, labels, method="logistic")
    assert len(out) == len(scores)
    for v in out:
        assert 0.0 < v < 1.0


def test_calibrate_piecewise_method() -> None:
    out = calibrate([0.1, 0.5, 0.9], [0, 1, 1], method="piecewise")
    assert out == pytest.approx([0.1, 0.5, 0.9])
