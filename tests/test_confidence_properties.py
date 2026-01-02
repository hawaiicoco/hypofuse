"""Property tests over seeded inputs for confidence module."""

from __future__ import annotations

import math
import random

from hypofuse.confidence import (
    expected_calibration_error,
    preserves_ranking,
    reliability_bins,
    synthetic_calibration_set,
    temperature_scale,
)


def test_ece_in_unit_interval() -> None:
    rng = random.Random(123)
    for _ in range(20):
        n = rng.randint(10, 200)
        confs = [rng.random() for _ in range(n)]
        accs = [float(rng.randint(0, 1)) for _ in range(n)]
        ece = expected_calibration_error(confs, accs, n_bins=5)
        assert 0.0 <= ece <= 1.0


def test_ece_synthetic_calibrated_below_bound() -> None:
    scores, labels, _ = synthetic_calibration_set(n=2000, seed=0, sharpness=1.0)
    float_labels = [float(y) for y in labels]
    ece = expected_calibration_error(scores, float_labels, n_bins=10)
    assert ece < 0.1


def test_temperature_t1_is_deterministic_softmax() -> None:
    scores = [1.0, 2.0, 3.0]
    result1 = temperature_scale(scores, temperature=1.0)
    result2 = temperature_scale(scores, temperature=1.0)
    # Exact float equality: same computation path
    assert result1 == result2
    # Verify it is the standard softmax
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps)
    expected = [e / z for e in exps]
    assert result1 == expected


def test_monotone_transforms_preserve_ranking() -> None:
    rng = random.Random(456)
    scores = sorted(rng.random() for _ in range(10))
    for temp in [0.5, 1.0, 2.0]:
        out = temperature_scale(scores, temperature=temp)
        assert preserves_ranking(scores, out)


def test_reliability_bins_counts_sum_to_n() -> None:
    rng = random.Random(789)
    n = 100
    confs = [rng.random() for _ in range(n)]
    accs = [float(rng.randint(0, 1)) for _ in range(n)]
    bins = reliability_bins(confs, accs, n_bins=7)
    assert sum(b.count for b in bins) == n
