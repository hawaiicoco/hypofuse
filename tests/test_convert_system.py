"""Tests for system_from_row / system_to_row."""

from __future__ import annotations

import json

import pytest

from hypofuse.convert import system_from_row, system_to_row
from hypofuse.exceptions import SchemaError
from hypofuse.manifests.system import SystemMetadata


def _row(**overrides: object) -> dict:
    base: dict = {
        "schema": "hypofuse.system",
        "system_id": "s1",
        "language": "en",
    }
    base.update(overrides)
    return base


def test_system_roundtrip() -> None:
    original = SystemMetadata(
        system_id="s1",
        language="en",
        vocabulary_size=5000,
        description="synth",
        acoustic_model="am",
        language_model="lm",
        decoder="beam",
        version="1.0",
    )
    row = system_to_row(original)
    restored = system_from_row(row)
    assert restored == original
    assert json.loads(json.dumps(row)) == row


def test_system_missing_required() -> None:
    row = _row()
    del row["language"]
    with pytest.raises(SchemaError, match="missing required field: language"):
        system_from_row(row)


def test_system_wrong_type() -> None:
    with pytest.raises(SchemaError, match="field vocabulary_size expected int"):
        system_from_row(_row(vocabulary_size="big"))


def test_system_unknown_key() -> None:
    with pytest.raises(SchemaError, match="unknown keys"):
        system_from_row(_row(extra="x"))


def test_system_defaults() -> None:
    obj = system_from_row(_row())
    assert obj.vocabulary_size == 0
    assert obj.description == ""
    assert obj.acoustic_model == ""
