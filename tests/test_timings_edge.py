"""Edge-case tests for empty and single-token tracks across all functions.

Synthetic data covering boundary conditions that individual module tests
may not exercise in combination.
"""

from __future__ import annotations

import pytest

from hypofuse.timings import (
    TimingTrack,
    TokenTiming,
    align_timings,
    duration_buckets,
    from_manifest_row,
    interpolate_gaps,
    snap_to_grid,
    timing_report,
    to_manifest_row,
)


def _t(token: str, start: float, end: float) -> TokenTiming:
    return TokenTiming(token=token, start_s=start, end_s=end)


def test_empty_track_through_all_transforms() -> None:
    empty = TimingTrack()
    assert snap_to_grid(empty).tokens == ()
    assert interpolate_gaps(empty, max_gap_s=1.0).tokens == ()
    assert to_manifest_row(empty) == {"tokens": [], "timings": []}
    steps = align_timings(empty, empty)
    assert steps == ()
    report = timing_report([empty])
    assert report.count == 1
    assert report.mean_s == pytest.approx(0.0)


def test_single_token_track_through_transforms() -> None:
    single = TimingTrack(tokens=(_t("only", 0.5, 1.5),))
    snapped = snap_to_grid(single, frame_s=0.01)
    assert len(snapped) == 1
    assert snapped.tokens[0].start_s == pytest.approx(0.5)
    assert snapped.tokens[0].end_s == pytest.approx(1.5)

    filled = interpolate_gaps(single, max_gap_s=1.0)
    assert filled.tokens == single.tokens

    row = to_manifest_row(single)
    restored = from_manifest_row(row)
    assert restored.tokens == single.tokens


def test_align_single_token_match() -> None:
    ref = TimingTrack(tokens=(_t("a", 0.0, 1.0),))
    hyp = TimingTrack(tokens=(_t("a", 0.1, 0.9),))
    steps = align_timings(ref, hyp)
    assert len(steps) == 1
    assert steps[0].op == "match"


def test_align_single_vs_empty() -> None:
    ref = TimingTrack(tokens=(_t("a", 0.0, 1.0),))
    hyp = TimingTrack()
    steps = align_timings(ref, hyp)
    assert len(steps) == 1
    assert steps[0].op == "deletion"


def test_duration_buckets_very_small() -> None:
    assert duration_buckets(0.001) == "<1.0s"


def test_duration_buckets_very_large() -> None:
    assert duration_buckets(9999.0) == ">=8.0s"


def test_timing_report_single_zero_duration_track() -> None:
    track = TimingTrack(tokens=(_t("a", 1.0, 1.0),))
    report = timing_report([track])
    assert report.count == 1
    assert report.mean_s == pytest.approx(0.0)
    assert report.gap_total_s == pytest.approx(0.0)


def test_interpolate_single_zero_duration_token() -> None:
    # A single zero-duration token with no neighbours: no silence to split
    track = TimingTrack(tokens=(_t("a", 1.0, 1.0),))
    result = interpolate_gaps(track, max_gap_s=10.0)
    assert result.tokens[0].duration_s == pytest.approx(0.0)


def test_snap_single_token_preserves_token_name() -> None:
    track = TimingTrack(tokens=(_t("hello", 0.123, 0.456),))
    snapped = snap_to_grid(track, frame_s=0.01)
    assert snapped.tokens[0].token == "hello"
