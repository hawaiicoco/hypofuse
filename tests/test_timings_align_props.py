"""Property-style tests for align_timings invariants."""

from __future__ import annotations

from hypofuse.alignment import edit_alignment
from hypofuse.timings import TimingTrack, TokenTiming, align_timings


def _t(token: str, start: float, end: float) -> TokenTiming:
    return TokenTiming(token=token, start_s=start, end_s=end)


def _track(*pairs: tuple[str, float, float]) -> TimingTrack:
    return TimingTrack(tokens=tuple(_t(t, s, e) for t, s, e in pairs))


def test_step_count_matches_edit_alignment_ops() -> None:
    ref = _track(("a", 0.0, 1.0), ("b", 1.0, 2.0), ("c", 2.0, 3.0))
    hyp = _track(("a", 0.0, 0.5), ("x", 0.5, 1.0), ("c", 2.0, 3.0), ("d", 3.0, 4.0))
    steps = align_timings(ref, hyp)
    alignment = edit_alignment(
        [tt.token for tt in ref.tokens],
        [tt.token for tt in hyp.tokens],
    )
    assert len(steps) == len(alignment.ops)


def test_ref_indices_monotone() -> None:
    ref = _track(("a", 0.0, 1.0), ("b", 1.0, 2.0), ("c", 2.0, 3.0))
    hyp = _track(("a", 0.0, 0.5), ("x", 0.5, 1.0), ("c", 2.0, 3.0))
    steps = align_timings(ref, hyp)
    ref_indices = [s.ref_index for s in steps if s.ref_index is not None]
    assert ref_indices == sorted(ref_indices)


def test_hyp_indices_monotone() -> None:
    ref = _track(("a", 0.0, 1.0), ("b", 1.0, 2.0), ("c", 2.0, 3.0))
    hyp = _track(("a", 0.0, 0.5), ("x", 0.5, 1.0), ("c", 2.0, 3.0))
    steps = align_timings(ref, hyp)
    hyp_indices = [s.hyp_index for s in steps if s.hyp_index is not None]
    assert hyp_indices == sorted(hyp_indices)


def test_covers_every_ref_position() -> None:
    ref = _track(("a", 0.0, 1.0), ("b", 1.0, 2.0), ("c", 2.0, 3.0))
    hyp = _track(("a", 0.0, 0.5), ("c", 2.0, 3.0))
    steps = align_timings(ref, hyp)
    ref_indices = {s.ref_index for s in steps if s.ref_index is not None}
    assert ref_indices == {0, 1, 2}


def test_covers_every_hyp_position() -> None:
    ref = _track(("a", 0.0, 1.0), ("c", 2.0, 3.0))
    hyp = _track(("a", 0.0, 0.5), ("x", 0.5, 1.0), ("c", 2.0, 3.0))
    steps = align_timings(ref, hyp)
    hyp_indices = {s.hyp_index for s in steps if s.hyp_index is not None}
    assert hyp_indices == {0, 1, 2}


def test_ref_and_hyp_lengths_via_alignment() -> None:
    ref = _track(("a", 0.0, 1.0), ("b", 1.0, 2.0))
    hyp = _track(("x", 0.0, 0.5), ("y", 0.5, 1.0), ("z", 1.0, 1.5))
    steps = align_timings(ref, hyp)
    ref_count = sum(1 for s in steps if s.ref_index is not None)
    hyp_count = sum(1 for s in steps if s.hyp_index is not None)
    assert ref_count == len(ref.tokens)
    assert hyp_count == len(hyp.tokens)
