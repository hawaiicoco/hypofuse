"""Temperature grid search without torch."""

from __future__ import annotations

import math

import pytest

from hypofuse.neural import fit_temperature, logistic_temperature_scale


def test_returns_finite_positive_value() -> None:
    scores = [0.2, 0.3, 0.7, 0.8, 0.9]
    labels = [0, 0, 1, 1, 1]
    t = fit_temperature(scores, labels)
    assert math.isfinite(t)
    assert t > 0


def test_custom_grid() -> None:
    scores = [0.1, 0.9, 0.1, 0.9]
    labels = [0, 1, 0, 1]
    t = fit_temperature(scores, labels, grid=[0.5, 1.0, 2.0])
    assert t in (0.5, 1.0, 2.0)


def test_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError, match="align"):
        fit_temperature([0.5], [0, 1])


def test_logistic_temperature_identity() -> None:
    # temperature=1 is the identity for logistic scaling
    scores = [0.2, 0.5, 0.8]
    scaled = logistic_temperature_scale(scores, temperature=1.0)
    for s, sc in zip(scores, scaled, strict=True):
        assert s == pytest.approx(sc, abs=1e-5)


def test_logistic_temperature_flattens() -> None:
    # High temperature pushes scores toward 0.5
    scores = [0.1, 0.9]
    scaled = logistic_temperature_scale(scores, temperature=10.0)
    assert abs(scaled[0] - 0.5) < 0.1
    assert abs(scaled[1] - 0.5) < 0.1


def test_logistic_temperature_invalid() -> None:
    with pytest.raises(ValueError, match="positive"):
        logistic_temperature_scale([0.5], temperature=0.0)
