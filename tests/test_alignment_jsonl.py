"""JSONL row export, import and roundtrip."""

from __future__ import annotations

import pytest

from hypofuse.alignment import (
    MATCH,
    SUB,
    alignment_from_jsonl_row,
    alignment_to_jsonl_row,
    edit_alignment,
)


def test_jsonl_golden_dict() -> None:
    align = edit_alignment(["a", "b"], ["a", "x"])
    row = alignment_to_jsonl_row(align, utterance_id="utt1", system="sysA")
    assert row == {
        "schema": "hypofuse.alignment",
        "schema_version": 1,
        "utterance_id": "utt1",
        "system": "sysA",
        "score": 1.0,
        "ref_length": 2,
        "hyp_length": 2,
        "errors": 1,
        "ops": [
            {"op": MATCH, "ref": "a", "hyp": "a"},
            {"op": SUB, "ref": "b", "hyp": "x"},
        ],
    }


def test_jsonl_roundtrip() -> None:
    align = edit_alignment(["a", "b", "c"], ["a", "x", "c"])
    row = alignment_to_jsonl_row(align, utterance_id="utt2")
    restored = alignment_from_jsonl_row(row)
    assert restored.ops == align.ops
    assert restored.score == pytest.approx(align.score)


def test_jsonl_rejects_unknown_schema() -> None:
    row = {
        "schema": "unknown.schema",
        "schema_version": 1,
        "utterance_id": "u",
        "system": "",
        "score": 0.0,
        "ref_length": 0,
        "hyp_length": 0,
        "errors": 0,
        "ops": [],
    }
    with pytest.raises(ValueError, match="unknown schema"):
        alignment_from_jsonl_row(row)


def test_jsonl_rejects_missing_field() -> None:
    row = {
        "schema": "hypofuse.alignment",
        "schema_version": 1,
        "utterance_id": "u",
        "system": "",
        "score": 0.0,
    }
    with pytest.raises(ValueError, match="missing"):
        alignment_from_jsonl_row(row)


def test_jsonl_default_system_is_empty() -> None:
    align = edit_alignment(["a"], ["a"])
    row = alignment_to_jsonl_row(align, utterance_id="u")
    assert row["system"] == ""
