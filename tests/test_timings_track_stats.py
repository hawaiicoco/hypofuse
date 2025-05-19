"""Tests for TimingTrack statistics: total_s, speaking_ratio, __len__."""

from __future__ import annotations

import pytest

from hypofuse.timings import TimingTrack, TokenTiming


def _t(token: str, start: float, end: float) -> TokenTiming:
    return TokenTiming(token=token, start_s=start, end_s=end)


def test_total_s_empty() -> None:
    assert TimingTrack().total_s == pytest.approx(0.0)


def test_total_s_single() -> None:
    track = TimingTrack(tokens=(_t("a", 0.5, 1.5),))
    assert track.total_s == pytest.approx(1.0)


def test_total_s_multi() -> None:
    # Span from first start (0.0) to last end (5.0)
    track = TimingTrack(tokens=(_t("a", 0.0, 1.0), _t("b", 3.0, 5.0)))
    assert track.total_s == pytest.approx(5.0)


def test_speaking_ratio_empty() -> None:
    assert TimingTrack().speaking_ratio() == pytest.approx(0.0)


def test_speaking_ratio_no_gaps() -> None:
    # Two touching tokens fill the entire span: 2.0 / 2.0 = 1.0
    track = TimingTrack(tokens=(_t("a", 0.0, 1.0), _t("b", 1.0, 2.0)))
    assert track.speaking_ratio() == pytest.approx(1.0)


def test_speaking_ratio_with_gaps() -> None:
    # Tokens: (0,1) and (2,3). Span = 3.0, speaking = 2.0.
    # Ratio = 2.0 / 3.0 = 0.6667
    track = TimingTrack(tokens=(_t("a", 0.0, 1.0), _t("b", 2.0, 3.0)))
    assert track.speaking_ratio() == pytest.approx(2.0 / 3.0)


def test_speaking_ratio_zero_span() -> None:
    # Single zero-duration token: span = 0, ratio = 0
    track = TimingTrack(tokens=(_t("a", 1.0, 1.0),))
    assert track.speaking_ratio() == pytest.approx(0.0)


def test_len_empty() -> None:
    assert len(TimingTrack()) == 0


def test_len_multi() -> None:
    track = TimingTrack(tokens=(_t("a", 0.0, 1.0), _t("b", 2.0, 3.0)))
    assert len(track) == 2
