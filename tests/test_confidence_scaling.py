"""Temperature and piecewise calibration behaviour."""

from __future__ import annotations

import pytest

from hypofuse.confidence import piecewise_calibrate, temperature_scale


def test_temperature_scale_sums_to_one() -> None:
    out = temperature_scale([0.0, 0.0, 0.0], temperature=1.0)
    assert sum(out) == pytest.approx(1.0)


def test_temperature_scale_zero_temperature_invalid() -> None:
    with pytest.raises(ValueError):
        temperature_scale([0.0, 1.0], temperature=0.0)


def test_temperature_scale_high_temperature_flattens() -> None:
    out = temperature_scale([10.0, 0.0], temperature=100.0)
    assert abs(out[0] - 0.5) < 0.05


def test_temperature_scale_low_temperature_sharpens() -> None:
    out = temperature_scale([10.0, 0.0], temperature=0.1)
    assert out[0] > 0.99


def test_piecewise_calibrate_returns_same_length() -> None:
    out = piecewise_calibrate([0.1, 0.5, 0.9], [0.0], [1.0], [0.0])
    assert len(out) == 3


def test_piecewise_calibrate_alignment_required() -> None:
    with pytest.raises(ValueError):
        piecewise_calibrate([0.1], [0.0, 0.5], [1.0], [0.0])
