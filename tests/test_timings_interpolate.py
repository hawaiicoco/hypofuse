"""Tests for interpolate_gaps."""

from __future__ import annotations

import pytest

from hypofuse.timings import TimingTrack, TokenTiming, interpolate_gaps


def _t(token: str, start: float, end: float) -> TokenTiming:
    return TokenTiming(token=token, start_s=start, end_s=end)


def _track(*pairs: tuple[str, float, float]) -> TimingTrack:
    return TimingTrack(tokens=tuple(_t(t, s, e) for t, s, e in pairs))


def test_zero_duration_token_gains_duration() -> None:
    # Synthetic track: (0,1), (2,2), (3,4)
    # Middle token "b" has zero duration.
    # prev_end=1.0, next_start=3.0, total_gap=2.0
    # new_start = (1.0 + 2.0)/2 = 1.5
    # new_end   = (2.0 + 3.0)/2 = 2.5
    track = _track(("a", 0.0, 1.0), ("b", 2.0, 2.0), ("c", 3.0, 4.0))
    result = interpolate_gaps(track, max_gap_s=10.0)
    mid = result.tokens[1]
    assert mid.duration_s > 0.0
    assert mid.start_s == pytest.approx(1.5)
    assert mid.end_s == pytest.approx(2.5)


def test_span_unchanged() -> None:
    track = _track(("a", 0.0, 1.0), ("b", 2.0, 2.0), ("c", 3.0, 4.0))
    result = interpolate_gaps(track, max_gap_s=10.0)
    assert result.total_s == pytest.approx(track.total_s)


def test_negative_max_gap_raises() -> None:
    track = _track(("a", 0.0, 1.0))
    with pytest.raises(ValueError, match="negative"):
        interpolate_gaps(track, max_gap_s=-1.0)


def test_gap_exceeding_max_is_skipped() -> None:
    # total_gap = 2.0 > max_gap_s = 1.0, so zero-duration token stays
    track = _track(("a", 0.0, 1.0), ("b", 2.0, 2.0), ("c", 3.0, 4.0))
    result = interpolate_gaps(track, max_gap_s=1.0)
    assert result.tokens[1].duration_s == pytest.approx(0.0)


def test_empty_track_unchanged() -> None:
    result = interpolate_gaps(TimingTrack(), max_gap_s=5.0)
    assert result.tokens == ()


def test_no_zero_duration_tokens_unchanged() -> None:
    track = _track(("a", 0.0, 1.0), ("b", 2.0, 3.0))
    result = interpolate_gaps(track, max_gap_s=10.0)
    assert result.tokens == track.tokens


def test_token_order_preserved() -> None:
    track = _track(("a", 0.0, 1.0), ("b", 2.0, 2.0), ("c", 3.0, 4.0))
    result = interpolate_gaps(track, max_gap_s=10.0)
    tokens = [tt.token for tt in result.tokens]
    assert tokens == ["a", "b", "c"]


def test_first_token_zero_duration() -> None:
    # First token zero-duration: prev_end = tt.start_s, so total_gap
    # = next_start - start_s. new_start stays, new_end moves forward.
    track = _track(("a", 0.0, 0.0), ("b", 2.0, 3.0))
    result = interpolate_gaps(track, max_gap_s=10.0)
    first = result.tokens[0]
    assert first.start_s == pytest.approx(0.0)
    assert first.end_s == pytest.approx(1.0)
