"""Slow benchmark: logistic + temperature on 20k synthetic points."""

from __future__ import annotations

import math
import random
import time

import pytest

from hypofuse.confidence import (
    expected_calibration_error,
    fit_logistic,
    fit_temperature,
)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@pytest.mark.slow
def test_logistic_and_temperature_reduce_ece() -> None:
    rng = random.Random(2024)
    n = 20_000
    logits = [rng.gauss(0, 1.5) for _ in range(n)]
    labels = [1.0 if rng.random() < _sigmoid(2 * z - 0.5) else 0.0 for z in logits]
    raw_probs = [_sigmoid(z) for z in logits]
    raw_ece = expected_calibration_error(raw_probs, labels, n_bins=10)
    t0 = time.monotonic()
    cal = fit_logistic(logits, labels)
    cal_probs = cal.apply(logits)
    cal_ece = expected_calibration_error(cal_probs, labels, n_bins=10)
    t_fit = fit_temperature(logits, labels)
    temp_probs = [_sigmoid(z / t_fit) for z in logits]
    temp_ece = expected_calibration_error(temp_probs, labels, n_bins=10)
    elapsed = time.monotonic() - t0
    assert cal_ece < raw_ece
    assert temp_ece < raw_ece
    assert elapsed < 60.0


@pytest.mark.slow
def test_logistic_determinism_across_runs() -> None:
    rng1 = random.Random(99)
    scores = [rng1.gauss(0, 1) for _ in range(5000)]
    labels = [1.0 if rng1.random() < _sigmoid(z) else 0.0 for z in scores]
    cal1 = fit_logistic(scores, labels)
    cal2 = fit_logistic(scores, labels)
    assert cal1.coef == cal2.coef
    assert cal1.intercept == cal2.intercept
