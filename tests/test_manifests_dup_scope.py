"""Tests for the scope parameter on reject_duplicate_ids."""

from __future__ import annotations

import pytest

from hypofuse.exceptions import DuplicateIdError
from hypofuse.manifests.validate import reject_duplicate_ids


def _nbest(uid: str) -> dict:
    return {
        "schema": "hypofuse.nbest",
        "utterance_id": uid,
        "system": "a",
        "language": "en",
    }


def _ref(uid: str) -> dict:
    return {"schema": "hypofuse.reference", "utterance_id": uid, "text": "hi"}


def test_default_scope_allows_same_id_across_schemas() -> None:
    """Default (schema scope): same id in different schemas is fine."""
    rows = [_nbest("u1"), _ref("u1")]
    reject_duplicate_ids(rows)


def test_schema_scope_rejects_within_schema() -> None:
    rows = [_nbest("u1"), _nbest("u1")]
    with pytest.raises(DuplicateIdError, match="lines"):
        reject_duplicate_ids(rows, scope="schema")


def test_file_scope_rejects_across_schemas() -> None:
    rows = [_nbest("u1"), _ref("u1")]
    with pytest.raises(DuplicateIdError, match="lines"):
        reject_duplicate_ids(rows, scope="file")


def test_file_scope_allows_different_ids() -> None:
    rows = [_nbest("u1"), _ref("u2")]
    reject_duplicate_ids(rows, scope="file")


def test_scope_error_message_includes_lines() -> None:
    rows = [_nbest("u1"), _nbest("u2"), _nbest("u1")]
    with pytest.raises(DuplicateIdError, match="1 and 3"):
        reject_duplicate_ids(rows, scope="schema")


def test_unknown_scope_rejected() -> None:
    with pytest.raises(ValueError, match="unknown scope"):
        reject_duplicate_ids([_nbest("u1")], scope="bogus")
