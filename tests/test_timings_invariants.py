"""Property and invariant tests for timing transforms.

Uses :func:`hypofuse.util.seeded` for deterministic random tracks
(synthetic data, labelled as such).
"""

from __future__ import annotations

import pytest

from hypofuse.alignment import edit_alignment
from hypofuse.timings import (
    TimingTrack,
    TokenTiming,
    align_timings,
    interpolate_gaps,
    snap_to_grid,
)
from hypofuse.util import seeded


def _make_synthetic_track(rng: object, n: int) -> TimingTrack:
    """Build a synthetic track with n non-overlapping tokens.

    Uses the provided Random instance for deterministic generation.
    """
    import random as _random

    assert isinstance(rng, _random.Random)
    tokens: list[TokenTiming] = []
    cursor = 0.0
    for i in range(n):
        duration = rng.uniform(0.01, 0.5)
        gap = rng.uniform(0.0, 0.3)
        start = cursor + gap
        end = start + duration
        tokens.append(TokenTiming(token=f"tok{i}", start_s=start, end_s=end))
        cursor = end
    track = TimingTrack(tokens=tuple(tokens))
    track.validate()
    return track


def test_snap_to_grid_idempotent() -> None:
    """Snapping twice produces the same result as snapping once."""
    rng = seeded(42)
    track = _make_synthetic_track(rng, n=20)
    once = snap_to_grid(track, frame_s=0.01)
    twice = snap_to_grid(once, frame_s=0.01)
    for a, b in zip(once.tokens, twice.tokens, strict=True):
        assert a.start_s == pytest.approx(b.start_s)
        assert a.end_s == pytest.approx(b.end_s)


def test_interpolate_gaps_preserves_order() -> None:
    """Token order is unchanged after interpolation."""
    # Build a track with a zero-duration token in the middle
    tokens = [
        TokenTiming(token="a", start_s=0.0, end_s=1.0),
        TokenTiming(token="b", start_s=2.0, end_s=2.0),  # zero-duration
        TokenTiming(token="c", start_s=3.0, end_s=4.0),
    ]
    track = TimingTrack(tokens=tuple(tokens))
    result = interpolate_gaps(track, max_gap_s=10.0)
    original_tokens = [tt.token for tt in track.tokens]
    result_tokens = [tt.token for tt in result.tokens]
    assert result_tokens == original_tokens


def test_interpolate_gaps_preserves_span() -> None:
    """Total span is unchanged after interpolation (for middle zeros)."""
    tokens = [
        TokenTiming(token="a", start_s=0.0, end_s=1.0),
        TokenTiming(token="b", start_s=2.0, end_s=2.0),
        TokenTiming(token="c", start_s=3.0, end_s=5.0),
    ]
    track = TimingTrack(tokens=tuple(tokens))
    result = interpolate_gaps(track, max_gap_s=10.0)
    assert result.total_s == pytest.approx(track.total_s)


def test_align_timings_step_count_matches_edit_ops() -> None:
    """align_timings step count equals len(edit_alignment(...).ops)."""
    rng_a = seeded(7)
    rng_b = seeded(8)
    ref = _make_synthetic_track(rng_a, n=10)
    hyp = _make_synthetic_track(rng_b, n=8)
    steps = align_timings(ref, hyp)
    alignment = edit_alignment(
        [tt.token for tt in ref.tokens],
        [tt.token for tt in hyp.tokens],
    )
    assert len(steps) == len(alignment.ops)


def test_snap_preserves_token_strings() -> None:
    """Snapping does not alter token strings."""
    rng = seeded(123)
    track = _make_synthetic_track(rng, n=15)
    snapped = snap_to_grid(track, frame_s=0.01)
    for original, snapped_tt in zip(track.tokens, snapped.tokens, strict=True):
        assert original.token == snapped_tt.token


def test_interpolate_nonnegative_durations() -> None:
    """All durations remain non-negative after interpolation."""
    tokens = [
        TokenTiming(token="a", start_s=0.0, end_s=1.0),
        TokenTiming(token="b", start_s=2.0, end_s=2.0),
        TokenTiming(token="c", start_s=3.0, end_s=4.0),
    ]
    track = TimingTrack(tokens=tuple(tokens))
    result = interpolate_gaps(track, max_gap_s=10.0)
    for tt in result.tokens:
        assert tt.duration_s >= 0.0
