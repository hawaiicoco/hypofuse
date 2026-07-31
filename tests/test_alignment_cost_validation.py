"""Tests for non-numeric score rejection in alignment deserialization."""

from __future__ import annotations

import pytest

from hypofuse.alignment import alignment_from_jsonl_row


def _base_row() -> dict[str, object]:
    """Build a valid alignment JSONL row for testing."""
    return {
        "schema": "hypofuse.alignment",
        "schema_version": 1,
        "utterance_id": "u1",
        "system": "",
        "score": 2.0,
        "ref_length": 3,
        "hyp_length": 3,
        "errors": 1,
        "ops": [
            {"op": "MATCH", "ref": "a", "hyp": "a"},
            {"op": "SUB", "ref": "b", "hyp": "c"},
            {"op": "MATCH", "ref": "d", "hyp": "d"},
        ],
    }


def test_non_numeric_score_raises() -> None:
    """A non-numeric score value raises ValueError."""
    row = _base_row()
    row["score"] = "not-a-number"
    with pytest.raises(ValueError, match="numeric"):
        alignment_from_jsonl_row(row)


def test_integer_score_accepted() -> None:
    """An integer score is accepted."""
    row = _base_row()
    row["score"] = 3
    result = alignment_from_jsonl_row(row)
    assert result.score == 3.0


def test_float_score_accepted() -> None:
    """A float score is accepted."""
    row = _base_row()
    row["score"] = 2.5
    result = alignment_from_jsonl_row(row)
    assert result.score == 2.5
