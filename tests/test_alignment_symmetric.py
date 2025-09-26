"""Symmetric error rate: bounds and symmetry."""

from __future__ import annotations

import pytest

from hypofuse.alignment import symmetric_error_rate, word_error_rate


def test_symmetric_identical() -> None:
    assert symmetric_error_rate(["a", "b"], ["a", "b"]) == 0.0


def test_symmetric_both_empty() -> None:
    assert symmetric_error_rate([], []) == 0.0


def test_symmetric_one_empty() -> None:
    # ref=["a"], hyp=[]: errors=1, mean_len=0.5, rate=2.0
    assert symmetric_error_rate(["a"], []) == pytest.approx(2.0)


def test_symmetric_bounds() -> None:
    cases = [
        (["a", "b", "c"], ["a", "b", "c"]),
        (["a", "b"], ["x", "y"]),
        (["a"], ["a", "b", "c"]),
        ([], ["a", "b"]),
    ]
    for ref, hyp in cases:
        rate = symmetric_error_rate(ref, hyp)
        assert 0.0 <= rate <= 2.0 + 1e-9


def test_symmetric_is_symmetric() -> None:
    ref = ["a", "b", "c"]
    hyp = ["a", "x"]
    assert symmetric_error_rate(ref, hyp) == pytest.approx(symmetric_error_rate(hyp, ref))


def test_wer_not_symmetric_when_lengths_differ() -> None:
    ref = ["a", "b"]
    hyp = ["a"]
    # wer(ref, hyp) = 1/2, wer(hyp, ref) = 1/1
    assert word_error_rate(ref, hyp) != word_error_rate(hyp, ref)
