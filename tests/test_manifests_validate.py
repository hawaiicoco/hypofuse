"""Tests for hypofuse.manifests.validate."""

from __future__ import annotations

import pytest

from hypofuse.exceptions import DuplicateIdError, SchemaError
from hypofuse.manifests.validate import reject_duplicate_ids, validate_record


def _nbest(uid: str = "u1") -> dict:
    return {"schema": "hypofuse.nbest", "utterance_id": uid, "system": "a", "language": "en"}


def test_validate_nbest_ok() -> None:
    assert validate_record(_nbest()) == "hypofuse.nbest"


def test_validate_missing_schema() -> None:
    with pytest.raises(SchemaError, match="missing schema"):
        validate_record({"utterance_id": "u1"})


def test_validate_unknown_schema() -> None:
    with pytest.raises(SchemaError, match="unknown schema"):
        validate_record({"schema": "nope", "utterance_id": "u1"})


def test_validate_missing_required_string() -> None:
    with pytest.raises(SchemaError, match="missing required string field: utterance_id"):
        validate_record({"schema": "hypofuse.nbest", "system": "a", "language": "en"})


def test_validate_empty_string_rejected() -> None:
    with pytest.raises(SchemaError, match="must be non-empty"):
        validate_record(
            {"schema": "hypofuse.nbest", "utterance_id": "", "system": "a", "language": "en"}
        )


def test_validate_numeric_field_wrong_type() -> None:
    with pytest.raises(SchemaError, match="acoustic_log10 must be numeric"):
        validate_record({**_nbest(), "acoustic_log10": "0.0"})


def test_validate_numeric_field_bool_rejected() -> None:
    with pytest.raises(SchemaError, match="acoustic_log10 must be numeric"):
        validate_record({**_nbest(), "acoustic_log10": True})


def test_reject_duplicate_ids_passes_unique() -> None:
    rows = [_nbest("a"), _nbest("b"), _nbest("c")]
    reject_duplicate_ids(rows)


def test_reject_duplicate_ids_flags_duplicate() -> None:
    rows = [_nbest("a"), _nbest("a")]
    with pytest.raises(DuplicateIdError, match="duplicate"):
        reject_duplicate_ids(rows)


def test_reject_duplicate_ids_skips_missing_id() -> None:
    rows = [{"schema": "hypofuse.system", "system_id": "s1"}, {}]
    reject_duplicate_ids(rows, id_field="utterance_id")


def test_reject_duplicate_ids_allows_same_id_across_schemas() -> None:
    rows = [
        _nbest("a"),
        {"schema": "hypofuse.reference", "utterance_id": "a", "text": "hi"},
    ]
    reject_duplicate_ids(rows)
