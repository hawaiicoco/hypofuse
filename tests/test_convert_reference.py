"""Tests for reference_from_row / reference_to_row."""

from __future__ import annotations

import json

import pytest

from hypofuse.convert import reference_from_row, reference_to_row
from hypofuse.exceptions import SchemaError
from hypofuse.manifests.reference import ReferenceTranscript


def _row(**overrides: object) -> dict:
    base: dict = {
        "schema": "hypofuse.reference",
        "utterance_id": "u1",
        "text": "hello",
    }
    base.update(overrides)
    return base


def test_reference_roundtrip() -> None:
    original = ReferenceTranscript(
        utterance_id="u1",
        text="hello",
        speaker_id="spk1",
        speaker_group="A",
        duration_s=1.5,
        noise_db=10.0,
        intent_domain="greeting",
        tokens=("hello",),
    )
    row = reference_to_row(original)
    restored = reference_from_row(row)
    assert restored == original
    assert json.loads(json.dumps(row)) == row


def test_reference_missing_required() -> None:
    row = _row()
    del row["text"]
    with pytest.raises(SchemaError, match="missing required field: text"):
        reference_from_row(row)


def test_reference_wrong_type() -> None:
    with pytest.raises(SchemaError, match="field text expected str"):
        reference_from_row(_row(text=123))


def test_reference_unknown_key() -> None:
    with pytest.raises(SchemaError, match="unknown keys"):
        reference_from_row(_row(bogus="x"))


def test_reference_defaults() -> None:
    obj = reference_from_row(_row())
    assert obj.speaker_id == ""
    assert obj.speaker_group == ""
    assert obj.duration_s == 0.0
    assert obj.noise_db == 0.0
    assert obj.intent_domain == ""
    assert obj.tokens == ()
