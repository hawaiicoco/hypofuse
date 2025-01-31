"""Tests for read_manifest / write_manifest round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from hypofuse.exceptions import DuplicateIdError, SchemaError
from hypofuse.manifests.jsonl import read_manifest, write_manifest


def _row(uid: str) -> dict:
    return {"schema": "hypofuse.nbest", "utterance_id": uid, "system": "a", "language": "en"}


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    rows = [_row("u1"), _row("u2")]
    target = tmp_path / "manifest.jsonl"
    write_manifest(target, rows)
    assert read_manifest(target) == rows


def test_read_manifest_validates_schema(tmp_path: Path) -> None:
    target = tmp_path / "bad.jsonl"
    target.write_text('{"utterance_id": "u1", "system": "a", "language": "en"}\n')
    with pytest.raises(SchemaError, match="missing schema"):
        read_manifest(target)


def test_read_manifest_detects_duplicate_ids(tmp_path: Path) -> None:
    target = tmp_path / "dup.jsonl"
    target.write_text(
        '{"schema": "hypofuse.nbest", "utterance_id": "u1", "system": "a", "language": "en"}\n'
        '{"schema": "hypofuse.nbest", "utterance_id": "u1", "system": "b", "language": "en"}\n'
    )
    with pytest.raises(DuplicateIdError):
        read_manifest(target)


def test_write_manifest_rejects_unknown_schema(tmp_path: Path) -> None:
    target = tmp_path / "x.jsonl"
    with pytest.raises(SchemaError, match="unknown schema"):
        write_manifest(target, [{"schema": "fake", "utterance_id": "u1"}])
