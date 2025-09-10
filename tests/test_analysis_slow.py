"""Slow bootstrap test: bounded time and determinism.

Synthetic data only; no benchmark claims.
"""

from __future__ import annotations

import time

import pytest

from hypofuse.analysis import UtteranceScore, paired_bootstrap_ci


@pytest.mark.slow
def test_large_bootstrap_deterministic_and_bounded() -> None:
    n = 500
    scores_a = [
        UtteranceScore(f"u{i}", ("a", "b"), ("a", "b") if i % 3 else ("x", "b")) for i in range(n)
    ]
    scores_b = [
        UtteranceScore(f"u{i}", ("a", "b"), ("x", "y") if i % 2 else ("a", "y")) for i in range(n)
    ]
    start = time.monotonic()
    r1 = paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=5000, seed=99)
    elapsed = time.monotonic() - start
    r2 = paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=5000, seed=99)
    # Determinism
    assert r1.cer_ci_low == r2.cer_ci_low
    assert r1.cer_ci_high == r2.cer_ci_high
    assert r1.wer_ci_low == r2.wer_ci_low
    assert r1.wer_ci_high == r2.wer_ci_high
    # Bounded time: generous limit to avoid flaky CI
    assert elapsed < 60.0
