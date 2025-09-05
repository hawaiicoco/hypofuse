"""Empty slice handling: zero utterances must not divide by zero."""

from __future__ import annotations

from hypofuse.analysis import UtteranceScore, slice_by_field, slice_metrics


def test_empty_slice_from_field_filtering() -> None:
    """A speaker group present in the manifest but absent after filtering."""
    items = [
        UtteranceScore("u0", ("a",), ("a",), speaker_group="alpha"),
        UtteranceScore("u1", ("a",), ("a",), speaker_group="alpha"),
    ]
    slices = slice_by_field(items, "speaker_group")
    # Manually inject an empty slice simulating a filtered-out group
    slices["beta"] = []
    metrics = slice_metrics(slices)
    beta_metric = next(m for m in metrics if m.key == "beta")
    assert beta_metric.count == 0
    assert beta_metric.cer == 0.0
    assert beta_metric.wer == 0.0


def test_empty_dict_no_division_by_zero() -> None:
    metrics = slice_metrics({"ghost": []})
    assert len(metrics) == 1
    assert metrics[0].count == 0
    assert metrics[0].wer == 0.0
    assert metrics[0].cer == 0.0


def test_mixed_empty_and_nonempty() -> None:
    items = [UtteranceScore("u0", ("a", "b"), ("a", "b"))]
    slices = {"has_items": items, "empty": []}
    metrics = slice_metrics(slices)
    counts = {m.key: m.count for m in metrics}
    assert counts["has_items"] == 1
    assert counts["empty"] == 0
