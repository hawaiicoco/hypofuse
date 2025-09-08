"""Paired bootstrap p-value for system comparison."""

from __future__ import annotations

import pytest

from hypofuse.analysis import UtteranceScore, paired_bootstrap_p_value


def test_equivalent_systems_near_half() -> None:
    scores_a: list[UtteranceScore] = []
    scores_b: list[UtteranceScore] = []
    for i in range(100):
        if i < 50:
            scores_a.append(UtteranceScore(f"u{i}", ("a", "b"), ("a", "b")))
            scores_b.append(UtteranceScore(f"u{i}", ("a", "b"), ("x", "y")))
        else:
            scores_a.append(UtteranceScore(f"u{i}", ("a", "b"), ("x", "y")))
            scores_b.append(UtteranceScore(f"u{i}", ("a", "b"), ("a", "b")))
    p = paired_bootstrap_p_value(scores_a, scores_b, iterations=2000, seed=0)
    assert p == pytest.approx(0.5, abs=0.1)


def test_strictly_better_system_small_p() -> None:
    scores_a = [UtteranceScore(f"u{i}", ("a", "b"), ("a", "b")) for i in range(20)]
    scores_b = [UtteranceScore(f"u{i}", ("a", "b"), ("x", "y")) for i in range(20)]
    p = paired_bootstrap_p_value(scores_a, scores_b, iterations=1000, seed=0)
    assert p < 0.05


def test_determinism() -> None:
    scores_a = [UtteranceScore(f"u{i}", ("a",), ("a",)) for i in range(10)]
    scores_b = [UtteranceScore(f"u{i}", ("a",), ("x",)) for i in range(10)]
    p1 = paired_bootstrap_p_value(scores_a, scores_b, seed=42, iterations=500)
    p2 = paired_bootstrap_p_value(scores_a, scores_b, seed=42, iterations=500)
    assert p1 == p2


def test_zero_iterations_raises() -> None:
    scores = [UtteranceScore("u0", ("a",), ("a",))]
    with pytest.raises(ValueError, match="iterations"):
        paired_bootstrap_p_value(scores, scores, iterations=0)
