"""Tests for migrate_row and migrate_manifest."""

from __future__ import annotations

import pytest

from hypofuse.convert import migrate_manifest, migrate_row, nbest_from_row
from hypofuse.exceptions import SchemaError


def _v1_row() -> dict:
    return {
        "schema": "hypofuse.nbest",
        "utterance_id": "u1",
        "system": "a",
        "language": "en",
        "hypotheses": [
            {"rank": 1, "text": "hi", "tokens": ["hi"], "score": -1.5},
            {"rank": 2, "text": "hey", "tokens": ["hey"], "score": -2.3},
        ],
    }


def test_migrate_v1_to_v2() -> None:
    """v1 score field is renamed to acoustic_log10 and lm_log10 added."""
    v1 = _v1_row()
    v2 = migrate_row(v1, to_version=2)
    assert v2["schema_version"] == 2
    for h in v2["hypotheses"]:
        assert "score" not in h
        assert "acoustic_log10" in h
        assert "lm_log10" in h
    # acoustic_log10 gets the old score value
    assert v2["hypotheses"][0]["acoustic_log10"] == -1.5
    assert v2["hypotheses"][0]["lm_log10"] == 0.0


def test_migrate_v2_unchanged() -> None:
    """An already-v2 row is returned unchanged (copy)."""
    v2 = {
        "schema": "hypofuse.nbest",
        "schema_version": 2,
        "utterance_id": "u1",
        "system": "a",
        "language": "en",
        "hypotheses": [
            {
                "rank": 1,
                "text": "hi",
                "tokens": ["hi"],
                "acoustic_log10": -1.5,
                "lm_log10": 0.0,
            }
        ],
    }
    result = migrate_row(v2, to_version=2)
    assert result == v2


def test_migrate_v1_to_v2_to_dataclass() -> None:
    """v1 -> v2 -> dataclass roundtrip works."""
    v1 = _v1_row()
    v2 = migrate_row(v1, to_version=2)
    obj = nbest_from_row(v2)
    assert obj.hypotheses[0].acoustic_log10 == -1.5
    assert obj.hypotheses[0].lm_log10 == 0.0


def test_migrate_rejects_downgrade() -> None:
    v2 = {
        "schema": "hypofuse.nbest",
        "schema_version": 2,
        "utterance_id": "u1",
        "system": "a",
        "language": "en",
        "hypotheses": [],
    }
    with pytest.raises(SchemaError, match="cannot downgrade"):
        migrate_row(v2, to_version=1)


def test_migrate_unknown_schema() -> None:
    with pytest.raises(SchemaError, match="unknown schema"):
        migrate_row({"schema": "fake"}, to_version=2)


def test_migrate_manifest() -> None:
    rows = [_v1_row(), _v1_row()]
    rows[1]["utterance_id"] = "u2"
    migrated = migrate_manifest(rows, to_version=2)
    assert all(r["schema_version"] == 2 for r in migrated)


def test_migrate_non_nbest_passthrough() -> None:
    """Non-nbest schemas pass through with version stamped."""
    ref = {
        "schema": "hypofuse.reference",
        "utterance_id": "u1",
        "text": "hi",
    }
    result = migrate_row(ref, to_version=2)
    assert result["schema_version"] == 2
    assert result["text"] == "hi"
