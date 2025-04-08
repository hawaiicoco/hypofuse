"""Slice-by-key error analysis."""

from __future__ import annotations

from hypofuse.analysis import UtteranceScore, slice_by_duration, slice_by_field, slice_metrics


def _items() -> list[UtteranceScore]:
    return [
        UtteranceScore("u1", ("the", "cat"), ("the", "cat"), "A", -5.0, 0.5, "greeting"),
        UtteranceScore("u2", ("a", "dog"), ("a", "dog"), "A", 25.0, 2.0, "greeting"),
        UtteranceScore("u3", ("the", "fish"), ("a", "fish"), "B", 15.0, 5.0, "qa"),
        UtteranceScore("u4", ("hello",), ("hello",), "B", 35.0, 8.0, "qa"),
    ]


def test_slice_by_duration_returns_buckets() -> None:
    items = [
        *_items(),
        UtteranceScore("u_long", ("a", "b"), ("a", "b"), "C", 0.0, 12.0, "qa"),
    ]
    out = slice_by_duration(items)
    assert "all" in out
    assert any(k.startswith("<") for k in out)
    assert any(k.startswith(">") for k in out)


def test_duration_buckets_partition_items() -> None:
    items = _items()
    out = slice_by_duration(items)
    total = sum(len(v) for k, v in out.items() if k != "all")
    assert total == len(items)


def test_slice_by_noise_categorizes() -> None:
    out = slice_by_field(_items(), "noise")
    assert "quiet" in out
    assert "noisy" in out


def test_slice_by_speaker_group() -> None:
    out = slice_by_field(_items(), "speaker_group")
    assert out["A"] and out["B"]


def test_slice_metrics_includes_all() -> None:
    slices = slice_by_duration(_items())
    metrics = slice_metrics(slices, use_cer=False)
    assert any(m.key == "all" for m in metrics)
    for m in metrics:
        if m.count:
            assert m.wer >= 0.0


def test_slice_metrics_handles_empty() -> None:
    metrics = slice_metrics({"empty": []})
    assert metrics[0].count == 0
