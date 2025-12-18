"""Reliability curve dataclass and computation."""

from __future__ import annotations

import pytest

from hypofuse.confidence import ReliabilityCurve, reliability_curve


def test_reliability_curve_hand_built() -> None:
    # 10 points, 5 bins -> 2 per bin
    # bin 0 [0.0, 0.2): 0.05, 0.15 -> mean_pred=0.10, obs=0.0, count=2
    # bin 1 [0.2, 0.4): 0.25, 0.35 -> mean_pred=0.30, obs=0.0, count=2
    # bin 2 [0.4, 0.6): 0.45, 0.55 -> mean_pred=0.50, obs=0.5, count=2
    # bin 3 [0.6, 0.8): 0.65, 0.75 -> mean_pred=0.70, obs=1.0, count=2
    # bin 4 [0.8, 1.0): 0.85, 0.95 -> mean_pred=0.90, obs=1.0, count=2
    probs = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    curve = reliability_curve(probs, labels, bins=5)
    assert isinstance(curve, ReliabilityCurve)
    assert len(curve.bin_edges) == 6
    assert curve.bin_edges[0] == pytest.approx(0.0)
    assert curve.bin_edges[-1] == pytest.approx(1.0)
    assert curve.counts == (2, 2, 2, 2, 2)
    assert curve.mean_predicted[0] == pytest.approx(0.1)
    assert curve.observed_frequency[2] == pytest.approx(0.5)


def test_reliability_curve_empty_bins() -> None:
    # All points in bin 0; bins 1-4 empty with count=0 and obs=0.0
    curve = reliability_curve([0.05, 0.06], [0, 1], bins=5)
    assert curve.counts[0] == 2
    for i in range(1, 5):
        assert curve.counts[i] == 0
        assert curve.observed_frequency[i] == pytest.approx(0.0)


def test_reliability_curve_edge_probability() -> None:
    # 0.2 with 5 bins: int(0.2*5) = int(1.0) = 1 -> bin 1 (upper bin)
    curve = reliability_curve([0.2], [1], bins=5)
    assert curve.counts[0] == 0
    assert curve.counts[1] == 1


def test_reliability_curve_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match=r"same length"):
        reliability_curve([0.5], [1, 0])


def test_reliability_curve_invalid_bins() -> None:
    with pytest.raises(ValueError, match=r">= 1"):
        reliability_curve([0.5], [1], bins=0)
