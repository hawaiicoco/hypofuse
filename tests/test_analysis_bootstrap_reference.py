"""The bootstrap CI must match a brute-force reference estimator.

``paired_bootstrap_ci`` precomputes per-utterance rates and re-averages them
per replicate. This test re-implements the estimator the slow, obvious way
(recompute every rate inside every replicate) and asserts the two agree
exactly, so the optimization cannot silently change the statistic.
"""

from __future__ import annotations

import random
import statistics

import pytest

from hypofuse.alignment import character_error_rate, word_error_rate
from hypofuse.analysis import UtteranceScore, paired_bootstrap_ci

SCORES_A = [
    UtteranceScore("u0", ("a", "b", "c"), ("a", "b", "c")),
    UtteranceScore("u1", ("a", "b", "c"), ("x", "b", "c")),
    UtteranceScore("u2", ("a", "b"), ("a",)),
    UtteranceScore("u3", ("a", "b"), ("a", "b", "c")),
    UtteranceScore("u4", ("hello", "world"), ("hello", "word")),
    UtteranceScore("u5", ("hello", "world"), ("hello", "world")),
]
SCORES_B = [
    UtteranceScore("u0", ("a", "b", "c"), ("a", "x", "c")),
    UtteranceScore("u1", ("a", "b", "c"), ("a", "b", "c")),
    UtteranceScore("u2", ("a", "b"), ("b", "a")),
    UtteranceScore("u3", ("a", "b"), ("a", "b")),
    UtteranceScore("u4", ("hello", "world"), ("hell", "world")),
    UtteranceScore("u5", ("hello", "world"), ("hello", "worlds")),
]


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def _reference_ci(n_bootstrap: int, seed: int, confidence: float) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(SCORES_A)
    deltas: list[float] = []
    for _ in range(n_bootstrap):
        idxs = [rng.randrange(n) for _ in range(n)]
        a = statistics.fmean(
            character_error_rate("".join(s.reference), "".join(s.hypothesis))
            for s in (SCORES_A[i] for i in idxs)
        )
        b = statistics.fmean(
            character_error_rate("".join(s.reference), "".join(s.hypothesis))
            for s in (SCORES_B[i] for i in idxs)
        )
        deltas.append(b - a)
    alpha = (1 - confidence) / 2
    return _quantile(deltas, alpha), _quantile(deltas, 1 - alpha)


def test_matches_brute_force_reference() -> None:
    row = paired_bootstrap_ci(SCORES_A, SCORES_B, n_bootstrap=200, seed=7, confidence=0.9)
    lo, hi = _reference_ci(200, 7, 0.9)
    # The reference interpolates the quantile the other way round
    # (``lo * (1 - f) + hi * f``), so compare with a tight tolerance instead
    # of bit equality; the resampling and the rates must match exactly.
    assert row.cer_ci_low == pytest.approx(lo, rel=1e-12, abs=1e-15)
    assert row.cer_ci_high == pytest.approx(hi, rel=1e-12, abs=1e-15)


def test_word_rate_matches_brute_force_reference() -> None:
    row = paired_bootstrap_ci(SCORES_A, SCORES_B, n_bootstrap=200, seed=11)
    rng = random.Random(11)
    n = len(SCORES_A)
    deltas: list[float] = []
    for _ in range(200):
        idxs = [rng.randrange(n) for _ in range(n)]
        a = statistics.fmean(
            word_error_rate(list(s.reference), list(s.hypothesis))
            for s in (SCORES_A[i] for i in idxs)
        )
        b = statistics.fmean(
            word_error_rate(list(s.reference), list(s.hypothesis))
            for s in (SCORES_B[i] for i in idxs)
        )
        deltas.append(b - a)
    assert row.wer_ci_low == pytest.approx(_quantile(deltas, 0.025), rel=1e-12, abs=1e-15)
    assert row.wer_ci_high == pytest.approx(_quantile(deltas, 0.975), rel=1e-12, abs=1e-15)


def test_point_estimates_are_full_sample_means() -> None:
    row = paired_bootstrap_ci(SCORES_A, SCORES_B, n_bootstrap=10, seed=3)
    cer_a = statistics.fmean(
        character_error_rate("".join(s.reference), "".join(s.hypothesis)) for s in SCORES_A
    )
    cer_b = statistics.fmean(
        character_error_rate("".join(s.reference), "".join(s.hypothesis)) for s in SCORES_B
    )
    assert row.delta_cer == cer_b - cer_a
    assert row.n_pairs == len(SCORES_A)
