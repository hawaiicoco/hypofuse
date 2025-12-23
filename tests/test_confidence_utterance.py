"""Utterance confidence aggregation modes."""

from __future__ import annotations

import pytest

from hypofuse.confidence import utterance_confidence


def test_mean() -> None:
    # (0.8 + 0.6 + 0.4) / 3 = 0.6
    assert utterance_confidence([0.8, 0.6, 0.4], aggregation="mean") == pytest.approx(0.6)


def test_min() -> None:
    assert utterance_confidence([0.8, 0.6, 0.4], aggregation="min") == pytest.approx(0.4)


def test_geometric() -> None:
    # (0.8 * 0.6 * 0.4) ^ (1/3)
    expected = (0.8 * 0.6 * 0.4) ** (1 / 3)
    assert utterance_confidence([0.8, 0.6, 0.4], aggregation="geometric") == pytest.approx(expected)


def test_length_normalized() -> None:
    # product of all token confidences
    assert utterance_confidence([0.8, 0.6, 0.4], aggregation="length_normalized") == pytest.approx(
        0.8 * 0.6 * 0.4
    )


def test_ordering_min_le_geometric_le_mean() -> None:
    vals = [0.8, 0.6, 0.4]
    v_min = utterance_confidence(vals, aggregation="min")
    v_geo = utterance_confidence(vals, aggregation="geometric")
    v_mean = utterance_confidence(vals, aggregation="mean")
    assert v_min <= v_geo <= v_mean


def test_empty_raises() -> None:
    with pytest.raises(ValueError, match=r"empty"):
        utterance_confidence([], aggregation="mean")


def test_unknown_aggregation() -> None:
    with pytest.raises(ValueError, match=r"unknown aggregation"):
        utterance_confidence([0.5], aggregation="magic")


def test_single_token_all_equal() -> None:
    for agg in ("mean", "min", "geometric", "length_normalized"):
        assert utterance_confidence([0.7], aggregation=agg) == pytest.approx(0.7)
