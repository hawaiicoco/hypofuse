"""Rank preservation guarantees for calibrators."""

from __future__ import annotations

import pytest

from hypofuse.confidence import (
    LogisticCalibrator,
    piecewise_calibrate,
    preserves_ranking,
    temperature_scale,
)
from hypofuse.exceptions import CalibrationError


def test_preserves_ranking_identical() -> None:
    assert preserves_ranking([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])


def test_preserves_ranking_monotone_transform() -> None:
    assert preserves_ranking([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])


def test_preserves_ranking_inversion() -> None:
    assert not preserves_ranking([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])


def test_preserves_ranking_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match=r"same length"):
        preserves_ranking([1.0], [1.0, 2.0])


def test_temperature_preserves_ranking() -> None:
    scores = [5.0, 1.0, -2.0, 3.0]
    for temp in [0.5, 1.0, 2.0, 10.0]:
        calibrated = temperature_scale(scores, temperature=temp)
        assert preserves_ranking(scores, calibrated)


def test_piecewise_monotone_preserves_ranking() -> None:
    scores = [0.1, 0.4, 0.7, 0.9]
    out = piecewise_calibrate(scores, [0.0], [2.0], [0.1])
    assert preserves_ranking(scores, out)


def test_logistic_positive_coef_preserves_ranking() -> None:
    cal = LogisticCalibrator(coef=2.0, intercept=-1.0)
    scores = [0.1, 0.5, 0.9, 0.3]
    out = cal.apply(scores)
    assert preserves_ranking(scores, out)


def test_piecewise_non_monotone_negative_slope_raises() -> None:
    with pytest.raises(CalibrationError, match=r"monotone"):
        piecewise_calibrate([0.5], [0.0, 0.5], [1.0, -1.0], [0.0, 1.0])


def test_piecewise_non_monotone_decreasing_boundary_raises() -> None:
    # segment 0 at bp=0.5 gives 1.0*0.5 + 0.0 = 0.5
    # segment 1 at bp=0.5 gives 0.0*0.5 + 0.1 = 0.1 < 0.5
    with pytest.raises(CalibrationError, match=r"monotone"):
        piecewise_calibrate([0.3], [0.0, 0.5], [1.0, 0.0], [0.0, 0.1])


def test_non_monotone_values_do_not_preserve_ranking() -> None:
    assert not preserves_ranking([1.0, 2.0, 3.0], [1.0, 3.0, 2.0])
