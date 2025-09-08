"""Paired bootstrap CI: bracket, swap, and degenerate properties."""

from __future__ import annotations

import pytest

from hypofuse.analysis import UtteranceScore, paired_bootstrap_ci


def _systems(n: int = 30) -> tuple[list[UtteranceScore], list[UtteranceScore]]:
    a: list[UtteranceScore] = []
    b: list[UtteranceScore] = []
    for i in range(n):
        hyp_a = ("a", "b") if i % 3 == 0 else ("x", "b")
        hyp_b = ("x", "y") if i % 2 == 0 else ("a", "y")
        a.append(UtteranceScore(f"u{i}", ("a", "b"), hyp_a))
        b.append(UtteranceScore(f"u{i}", ("a", "b"), hyp_b))
    return a, b


def test_ci_brackets_point_estimate() -> None:
    a, b = _systems()
    row = paired_bootstrap_ci(a, b, n_bootstrap=500, seed=7)
    assert row.cer_ci_low <= row.delta_cer <= row.cer_ci_high
    assert row.wer_ci_low <= row.delta_wer <= row.wer_ci_high


def test_swap_negates_bounds() -> None:
    a, b = _systems()
    fwd = paired_bootstrap_ci(a, b, n_bootstrap=500, seed=7)
    rev = paired_bootstrap_ci(b, a, n_bootstrap=500, seed=7)
    assert rev.delta_wer == pytest.approx(-fwd.delta_wer)
    assert rev.wer_ci_low == pytest.approx(-fwd.wer_ci_high)
    assert rev.wer_ci_high == pytest.approx(-fwd.wer_ci_low)


def test_degenerate_all_equal_scores() -> None:
    scores_a = [UtteranceScore(f"u{i}", ("a", "b"), ("a", "b")) for i in range(20)]
    scores_b = [UtteranceScore(f"u{i}", ("a", "b"), ("a", "b")) for i in range(20)]
    row = paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=200, seed=0)
    assert row.delta_wer == 0.0
    assert row.wer_ci_low == 0.0
    assert row.wer_ci_high == 0.0
