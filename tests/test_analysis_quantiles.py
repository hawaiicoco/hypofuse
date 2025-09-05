"""Quantile-based slicing of utterance scores."""

from __future__ import annotations

from hypofuse.analysis import UtteranceScore, slice_by_quantiles


def _items() -> list[UtteranceScore]:
    return [
        UtteranceScore("u0", ("a",), ("a",), duration_s=1.0),
        UtteranceScore("u1", ("a",), ("a",), duration_s=2.0),
        UtteranceScore("u2", ("a",), ("a",), duration_s=3.0),
        UtteranceScore("u3", ("a",), ("a",), duration_s=4.0),
        UtteranceScore("u4", ("a",), ("a",), duration_s=5.0),
        UtteranceScore("u5", ("a",), ("a",), duration_s=6.0),
        UtteranceScore("u6", ("a",), ("a",), duration_s=7.0),
        UtteranceScore("u7", ("a",), ("a",), duration_s=8.0),
    ]


def test_every_score_in_exactly_one_slice() -> None:
    items = _items()
    result = slice_by_quantiles(items, "duration_s", buckets=4)
    all_ids: list[str] = []
    for group in result.values():
        for it in group:
            all_ids.append(it.utterance_id)
    assert sorted(all_ids) == sorted(it.utterance_id for it in items)
    assert len(all_ids) == len(set(all_ids))


def test_all_equal_field_yields_single_slice() -> None:
    items = [UtteranceScore(f"u{i}", ("a",), ("a",), duration_s=5.0) for i in range(5)]
    result = slice_by_quantiles(items, "duration_s", buckets=4)
    assert len(result) == 1
    assert "all" in result
    assert len(result["all"]) == 5


def test_empty_input() -> None:
    result = slice_by_quantiles([], "duration_s")
    assert result == {}


def test_single_bucket() -> None:
    items = _items()
    result = slice_by_quantiles(items, "duration_s", buckets=1)
    total = sum(len(v) for v in result.values())
    assert total == len(items)
