"""Tests for snap_to_grid frame quantization."""

from __future__ import annotations

import pytest

from hypofuse.timings import TimingTrack, TokenTiming, snap_to_grid


def _t(token: str, start: float, end: float) -> TokenTiming:
    return TokenTiming(token=token, start_s=start, end_s=end)


def _track(*pairs: tuple[str, float, float]) -> TimingTrack:
    return TimingTrack(tokens=tuple(_t(t, s, e) for t, s, e in pairs))


def test_hand_computed_example() -> None:
    # frame_s = 0.01
    # start_s = 0.015 -> 0.015/0.01 = 1.5 -> round-half-away = 2 -> 0.02
    # end_s   = 0.025 -> 0.025/0.01 = 2.5 -> round-half-away = 3 -> 0.03
    track = _track(("a", 0.015, 0.025))
    result = snap_to_grid(track, frame_s=0.01)
    assert result.tokens[0].start_s == pytest.approx(0.02)
    assert result.tokens[0].end_s == pytest.approx(0.03)


def test_already_on_grid() -> None:
    track = _track(("a", 0.01, 0.05))
    result = snap_to_grid(track, frame_s=0.01)
    assert result.tokens[0].start_s == pytest.approx(0.01)
    assert result.tokens[0].end_s == pytest.approx(0.05)


def test_no_negative_boundaries() -> None:
    # A token very close to zero should snap to 0.0, not a negative value
    track = _track(("a", 0.001, 0.009))
    result = snap_to_grid(track, frame_s=0.01)
    assert result.tokens[0].start_s >= 0.0
    assert result.tokens[0].end_s >= 0.0


def test_no_inverted_intervals() -> None:
    # A very short token that would invert after snapping
    # start=0.004 -> 0.0, end=0.004 -> 0.0 => clamped to (0.0, 0.0)
    track = _track(("a", 0.004, 0.004))
    result = snap_to_grid(track, frame_s=0.01)
    assert result.tokens[0].end_s >= result.tokens[0].start_s


def test_empty_track() -> None:
    result = snap_to_grid(TimingTrack(), frame_s=0.01)
    assert result.tokens == ()


def test_invalid_frame_s() -> None:
    track = _track(("a", 0.0, 1.0))
    with pytest.raises(ValueError, match="positive"):
        snap_to_grid(track, frame_s=0.0)
    with pytest.raises(ValueError, match="positive"):
        snap_to_grid(track, frame_s=-0.01)


def test_round_half_away_at_boundary() -> None:
    # 0.005 / 0.01 = 0.5 -> round-half-away = 1 -> 0.01
    track = _track(("a", 0.005, 0.015))
    result = snap_to_grid(track, frame_s=0.01)
    assert result.tokens[0].start_s == pytest.approx(0.01)
    # 0.015 / 0.01 = 1.5 -> round-half-away = 2 -> 0.02
    assert result.tokens[0].end_s == pytest.approx(0.02)
