"""Tests for validate_manifest_file."""

from __future__ import annotations

from pathlib import Path

import pytest

from hypofuse.exceptions import DuplicateIdError, SchemaError
from hypofuse.manifests.validate import ManifestSummary, validate_manifest_file


def _write(tmp_path: Path, name: str, lines: list[str]) -> Path:
    target = tmp_path / name
    target.write_text("".join(lines), encoding="utf-8")
    return target


def test_valid_file(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "ok.jsonl",
        [
            '{"schema": "hypofuse.nbest", "utterance_id": "u1", "system": "a", "language": "en"}\n',
            '{"schema": "hypofuse.reference", "utterance_id": "u1", "text": "hi"}\n',
        ],
    )
    summary = validate_manifest_file(path)
    assert isinstance(summary, ManifestSummary)
    assert summary.total_rows == 2
    assert summary.counts_by_schema["hypofuse.nbest"] == 1
    assert summary.counts_by_schema["hypofuse.reference"] == 1


def test_duplicate_id_is_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "dup.jsonl",
        [
            '{"schema": "hypofuse.nbest", "utterance_id": "u1", "system": "a", "language": "en"}\n',
            '{"schema": "hypofuse.nbest", "utterance_id": "u1", "system": "b", "language": "en"}\n',
        ],
    )
    # The exact wording changes with the scope work; the id must be named.
    with pytest.raises(DuplicateIdError, match="duplicate"):
        validate_manifest_file(path)


def test_bad_row_names_line(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "bad.jsonl",
        [
            '{"schema": "hypofuse.nbest", "utterance_id": "u1", "system": "a", "language": "en"}\n',
            '{"utterance_id": "u2"}\n',
        ],
    )
    with pytest.raises(SchemaError, match="line"):
        validate_manifest_file(path)


def test_escaping_audio_path_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "escape.jsonl",
        [
            '{"schema": "hypofuse.nbest", "utterance_id": "u1",'
            ' "system": "a", "language": "en",'
            ' "audio_path": "../secret.wav"}\n',
        ],
    )
    with pytest.raises(SchemaError, match="parent-directory"):
        validate_manifest_file(path)


def test_require_schemas(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "partial.jsonl",
        [
            '{"schema": "hypofuse.nbest", "utterance_id": "u1", "system": "a", "language": "en"}\n',
        ],
    )
    with pytest.raises(SchemaError, match="missing required schemas"):
        validate_manifest_file(path, require_schemas=["hypofuse.nbest", "hypofuse.reference"])
