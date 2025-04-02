"""Calibration is rank-preserving under monotonic transforms."""

from __future__ import annotations

from hypofuse.confidence import piecewise_calibrate, temperature_scale


def test_temperature_preserves_argmax() -> None:
    scores = [5.0, 1.0, -2.0]
    calibrated = temperature_scale(scores, temperature=1.5)
    assert calibrated[0] > calibrated[1] > calibrated[2]


def test_piecewise_monotone_preserves_order() -> None:
    scores = [0.1, 0.4, 0.7, 0.9]
    # Identity transform.
    out = piecewise_calibrate(scores, [0.0], [1.0], [0.0])
    assert out == scores


def test_piecewise_strictly_increasing_preserves_order() -> None:
    scores = [0.1, 0.4, 0.7, 0.9]
    # Linear transform y = 2x + 0.1.
    out = piecewise_calibrate(scores, [0.0], [2.0], [0.1])
    assert out == sorted(out)
