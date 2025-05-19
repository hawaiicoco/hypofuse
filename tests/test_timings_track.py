"""Tests for :class:`hypofuse.timings.TimingTrack` validation."""

from __future__ import annotations

import pytest

from hypofuse.timings import TimingTrack, TokenTiming


def _t(token: str, start: float, end: float) -> TokenTiming:
    return TokenTiming(token=token, start_s=start, end_s=end)


def test_valid_track_passes() -> None:
    track = TimingTrack(tokens=(_t("a", 0.0, 1.0), _t("b", 1.0, 2.0)))
    track.validate()


def test_touching_boundaries_allowed() -> None:
    track = TimingTrack(tokens=(_t("a", 0.0, 1.0), _t("b", 1.0, 2.0)))
    track.validate()  # end == next start is fine


def test_overlap_rejected() -> None:
    track = TimingTrack(tokens=(_t("a", 0.0, 1.0), _t("b", 0.5, 2.0)))
    with pytest.raises(ValueError, match="overlapping"):
        track.validate()


def test_tolerance_allows_tiny_overlap() -> None:
    # Overlap of 5e-10 is within default tolerance of 1e-9
    track = TimingTrack(tokens=(_t("a", 0.0, 1.0), _t("b", 1.0 - 5e-10, 2.0)))
    track.validate()


def test_tolerance_rejects_large_overlap() -> None:
    # Overlap of 2e-9 exceeds default tolerance of 1e-9
    track = TimingTrack(tokens=(_t("a", 0.0, 1.0), _t("b", 1.0 - 2e-9, 2.0)))
    with pytest.raises(ValueError, match="overlapping"):
        track.validate()


def test_nonfinite_rejected() -> None:
    track = TimingTrack(tokens=(_t("a", 0.0, float("inf")),))
    with pytest.raises(ValueError, match="non-finite"):
        track.validate()


def test_nan_in_token_rejected() -> None:
    # TokenTiming.validate catches NaN before TimingTrack checks isfinite
    track = TimingTrack(tokens=(_t("a", float("nan"), 1.0),))
    with pytest.raises(ValueError, match="NaN"):
        track.validate()


def test_empty_track_valid() -> None:
    track = TimingTrack()
    track.validate()


def test_frozen() -> None:
    from dataclasses import FrozenInstanceError

    track = TimingTrack()
    with pytest.raises(FrozenInstanceError):
        track.tokens = ()  # type: ignore[misc]
