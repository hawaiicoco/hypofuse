"""Tests for align_timings basic behaviour."""

from __future__ import annotations

import pytest

from hypofuse.timings import (
    TimingTrack,
    TokenTiming,
    align_timings,
)


def _t(token: str, start: float, end: float) -> TokenTiming:
    return TokenTiming(token=token, start_s=start, end_s=end)


def _track(*pairs: tuple[str, float, float]) -> TimingTrack:
    return TimingTrack(tokens=tuple(_t(t, s, e) for t, s, e in pairs))


def test_perfect_match() -> None:
    ref = _track(("a", 0.0, 1.0), ("b", 1.0, 2.0))
    hyp = _track(("a", 0.1, 0.9), ("b", 1.1, 1.9))
    steps = align_timings(ref, hyp)
    assert len(steps) == 2
    assert all(s.op == "match" for s in steps)
    # Timings come from hypothesis
    assert steps[0].start_s == pytest.approx(0.1)
    assert steps[0].end_s == pytest.approx(0.9)


def test_substitution() -> None:
    ref = _track(("a", 0.0, 1.0), ("b", 1.0, 2.0))
    hyp = _track(("a", 0.0, 1.0), ("x", 1.0, 2.0))
    steps = align_timings(ref, hyp)
    ops = [s.op for s in steps]
    assert ops == ["match", "substitution"]


def test_insertion() -> None:
    ref = _track(("a", 0.0, 1.0), ("b", 1.0, 2.0))
    hyp = _track(("a", 0.0, 0.5), ("x", 0.5, 1.0), ("b", 1.0, 2.0))
    steps = align_timings(ref, hyp)
    ops = [s.op for s in steps]
    assert ops == ["match", "insertion", "match"]
    ins = steps[1]
    assert ins.ref_index is None
    assert ins.hyp_index == 1


def test_deletion() -> None:
    ref = _track(("a", 0.0, 1.0), ("b", 1.0, 2.0), ("c", 2.0, 3.0))
    hyp = _track(("a", 0.0, 1.0), ("c", 2.0, 3.0))
    steps = align_timings(ref, hyp)
    ops = [s.op for s in steps]
    assert ops == ["match", "deletion", "match"]
    dele = steps[1]
    assert dele.hyp_index is None
    assert dele.ref_index == 1
    # Deletion timings come from reference
    assert dele.start_s == pytest.approx(1.0)
    assert dele.end_s == pytest.approx(2.0)


def test_both_empty() -> None:
    steps = align_timings(TimingTrack(), TimingTrack())
    assert steps == ()


def test_ref_empty_hyp_nonempty() -> None:
    hyp = _track(("a", 0.0, 1.0))
    steps = align_timings(TimingTrack(), hyp)
    assert len(steps) == 1
    assert steps[0].op == "insertion"


def test_indices_are_correct() -> None:
    ref = _track(("a", 0.0, 1.0), ("b", 1.0, 2.0))
    hyp = _track(("a", 0.0, 0.5), ("b", 0.5, 1.0))
    steps = align_timings(ref, hyp)
    assert steps[0].ref_index == 0
    assert steps[0].hyp_index == 0
    assert steps[1].ref_index == 1
    assert steps[1].hyp_index == 1
