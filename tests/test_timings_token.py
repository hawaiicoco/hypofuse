"""Tests for :class:`hypofuse.timings.TokenTiming`."""

from __future__ import annotations

import pytest

from hypofuse.timings import TokenTiming


def test_duration_s() -> None:
    tt = TokenTiming(token="hello", start_s=0.5, end_s=1.5)
    assert tt.duration_s == pytest.approx(1.0)


def test_duration_s_zero() -> None:
    tt = TokenTiming(token="x", start_s=2.0, end_s=2.0)
    assert tt.duration_s == pytest.approx(0.0)


def test_validate_passes_for_valid() -> None:
    tt = TokenTiming(token="a", start_s=0.0, end_s=1.0)
    tt.validate()  # should not raise


def test_validate_rejects_end_before_start() -> None:
    tt = TokenTiming(token="a", start_s=2.0, end_s=1.0)
    with pytest.raises(ValueError, match="end_s must not precede start_s"):
        tt.validate()


def test_validate_rejects_nan_start() -> None:
    tt = TokenTiming(token="a", start_s=float("nan"), end_s=1.0)
    with pytest.raises(ValueError, match="NaN"):
        tt.validate()


def test_validate_rejects_nan_end() -> None:
    tt = TokenTiming(token="a", start_s=0.0, end_s=float("nan"))
    with pytest.raises(ValueError, match="NaN"):
        tt.validate()


def test_validate_rejects_negative_start() -> None:
    tt = TokenTiming(token="a", start_s=-0.1, end_s=1.0)
    with pytest.raises(ValueError, match="negative"):
        tt.validate()


def test_validate_rejects_negative_end() -> None:
    tt = TokenTiming(token="a", start_s=0.0, end_s=-1.0)
    with pytest.raises(ValueError, match="negative"):
        tt.validate()


def test_frozen() -> None:
    from dataclasses import FrozenInstanceError

    tt = TokenTiming(token="a", start_s=0.0, end_s=1.0)
    with pytest.raises(FrozenInstanceError):
        tt.token = "b"  # type: ignore[misc]


def test_touching_boundary_is_valid() -> None:
    # end_s == start_s is a zero-duration token, which is valid
    tt = TokenTiming(token="a", start_s=1.0, end_s=1.0)
    tt.validate()
