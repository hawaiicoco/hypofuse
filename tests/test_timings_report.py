"""Tests for timing_report summary statistics.

Hand-computed reference values (synthetic data):

Tracks with total_s = [1.0, 2.0, 3.0, 4.0, 5.0]:
  mean   = 15.0 / 5 = 3.0
  median = nearest-rank(0.5, n=5) = sorted[ceil(0.5*5)-1] = sorted[2] = 3.0
  p95    = nearest-rank(0.95, n=5) = sorted[ceil(0.95*5)-1] = sorted[4] = 5.0

Gaps (inter-token silence per track):
  Track 1: (0.0, 0.5) (0.6, 1.0) -> gap = 0.1
  Track 2: (0.0, 2.0)             -> no gap
  Track 3: (0.0, 1.0) (1.5, 3.0) -> gap = 0.5
  Track 4: (0.0, 4.0)             -> no gap
  Track 5: (0.0, 2.5) (3.0, 5.0) -> gap = 0.5
  Total gap = 0.1 + 0.0 + 0.5 + 0.0 + 0.5 = 1.1
"""

from __future__ import annotations

import pytest

from hypofuse.timings import (
    TimingReport,
    TimingTrack,
    TokenTiming,
    timing_report,
)


def _t(token: str, start: float, end: float) -> TokenTiming:
    return TokenTiming(token=token, start_s=start, end_s=end)


def _track(*pairs: tuple[str, float, float]) -> TimingTrack:
    return TimingTrack(tokens=tuple(_t(t, s, e) for t, s, e in pairs))


# Synthetic 5-track fixture with known total_s and gaps.
_TRACKS = [
    _track(("a", 0.0, 0.5), ("b", 0.6, 1.0)),  # total_s=1.0, gap=0.1
    _track(("a", 0.0, 2.0)),  # total_s=2.0, gap=0
    _track(("a", 0.0, 1.0), ("b", 1.5, 3.0)),  # total_s=3.0, gap=0.5
    _track(("a", 0.0, 4.0)),  # total_s=4.0, gap=0
    _track(("a", 0.0, 2.5), ("b", 3.0, 5.0)),  # total_s=5.0, gap=0.5
]


def test_count() -> None:
    report = timing_report(_TRACKS)
    assert report.count == 5


def test_mean() -> None:
    report = timing_report(_TRACKS)
    assert report.mean_s == pytest.approx(3.0)


def test_median() -> None:
    report = timing_report(_TRACKS)
    # nearest-rank(0.5, n=5): ceil(2.5) = 3, index 2 -> 3.0
    assert report.median_s == pytest.approx(3.0)


def test_p95() -> None:
    report = timing_report(_TRACKS)
    # nearest-rank(0.95, n=5): ceil(4.75) = 5, index 4 -> 5.0
    assert report.p95_s == pytest.approx(5.0)


def test_gap_total() -> None:
    report = timing_report(_TRACKS)
    assert report.gap_total_s == pytest.approx(1.1)


def test_empty_tracks() -> None:
    report = timing_report([])
    assert report == TimingReport(count=0, mean_s=0.0, median_s=0.0, p95_s=0.0, gap_total_s=0.0)


def test_single_track() -> None:
    report = timing_report([_track(("a", 0.0, 2.0))])
    assert report.count == 1
    assert report.mean_s == pytest.approx(2.0)
    assert report.median_s == pytest.approx(2.0)
    assert report.p95_s == pytest.approx(2.0)
    assert report.gap_total_s == pytest.approx(0.0)


def test_report_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    report = timing_report(_TRACKS)
    with pytest.raises(FrozenInstanceError):
        report.count = 99  # type: ignore[misc]
