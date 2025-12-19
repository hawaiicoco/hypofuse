"""Temperature fitting by deterministic grid search."""

from __future__ import annotations

import math
import random

import pytest

from hypofuse.confidence import calibrate, fit_temperature


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _nll(scores: list[float], labels: list[float], temp: float) -> float:
    total = 0.0
    eps = 1e-15
    for s, y in zip(scores, labels, strict=True):
        p = max(eps, min(1.0 - eps, _sigmoid(s / temp)))
        total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
    return total


def test_fit_temperature_recovers_true() -> None:
    t_true = 2.0
    rng = random.Random(0)
    logits = [rng.gauss(0, 1) for _ in range(2000)]
    scores = [z * t_true for z in logits]
    labels = [1.0 if rng.random() < _sigmoid(z) else 0.0 for z in logits]
    t_fit = fit_temperature(scores, labels)
    assert abs(t_fit - t_true) <= 0.05


def test_fit_temperature_no_worse_than_identity() -> None:
    rng = random.Random(7)
    scores = [rng.gauss(0, 1) for _ in range(500)]
    labels = [1.0 if rng.random() < _sigmoid(z) else 0.0 for z in scores]
    t_fit = fit_temperature(scores, labels)
    assert _nll(scores, labels, t_fit) <= _nll(scores, labels, 1.0) + 1e-12


def test_fit_temperature_invalid_grid_empty() -> None:
    with pytest.raises(ValueError, match=r"empty"):
        fit_temperature([1.0], [1.0], grid=[])


def test_fit_temperature_invalid_grid_non_positive() -> None:
    with pytest.raises(ValueError, match=r"positive"):
        fit_temperature([1.0], [1.0], grid=[-1.0])


def test_calibrate_temperature_method() -> None:
    rng = random.Random(99)
    logits = [rng.gauss(0, 1) for _ in range(500)]
    labels = [1.0 if rng.random() < _sigmoid(z) else 0.0 for z in logits]
    sharp = [z * 3 for z in logits]
    out = calibrate(sharp, labels, method="temperature")
    assert len(out) == len(sharp)
    for v in out:
        assert 0.0 < v < 1.0
