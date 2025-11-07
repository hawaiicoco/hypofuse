"""Tests for atomic write and byte-level output guarantees."""

from __future__ import annotations

from pathlib import Path

import pytest

from hypofuse.exceptions import SchemaError
from hypofuse.manifests.jsonl import write_manifest


def _row(uid: str) -> dict:
    return {
        "schema": "hypofuse.nbest",
        "utterance_id": uid,
        "system": "a",
        "language": "en",
    }


def test_atomic_no_partial_file_on_error(tmp_path: Path) -> None:
    """A non-serializable row prevents any file from being created."""
    target = tmp_path / "fail.jsonl"
    bad_rows: list = [
        _row("u1"),
        {
            "schema": "hypofuse.nbest",
            "utterance_id": "u2",
            "system": "a",
            "language": "en",
            "bad": object(),
        },
    ]
    with pytest.raises((SchemaError, TypeError)):
        write_manifest(target, bad_rows)
    assert not target.exists()


def test_write_utf8_trailing_newline(tmp_path: Path) -> None:
    target = tmp_path / "ok.jsonl"
    write_manifest(target, [_row("u1")])
    raw = target.read_bytes()
    assert raw.endswith(b"\n")
    # UTF-8 encoding
    text = raw.decode("utf-8")
    assert text.count("\n") == 1


def test_write_one_object_per_line(tmp_path: Path) -> None:
    target = tmp_path / "multi.jsonl"
    rows = [_row("u1"), _row("u2"), _row("u3")]
    write_manifest(target, rows)
    lines = target.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    # Each line is valid JSON
    import json

    for line in lines:
        json.loads(line)


def test_write_ensure_ascii_false(tmp_path: Path) -> None:
    """Non-ASCII characters are written as-is, not escaped."""
    target = tmp_path / "unicode.jsonl"
    rows = [
        {
            "schema": "hypofuse.reference",
            "utterance_id": "u1",
            "text": "\u00e9",  # e-acute
        }
    ]
    write_manifest(target, rows)
    raw = target.read_bytes()
    # \u00e9 in UTF-8 is 0xc3 0xa9 -- both bytes should appear raw
    assert b"\xc3\xa9" in raw


def test_write_compact_json(tmp_path: Path) -> None:
    """Output is compact (no extra whitespace in JSON)."""
    target = tmp_path / "compact.jsonl"
    write_manifest(target, [_row("u1")])
    line = target.read_text(encoding="utf-8").strip()
    # compact JSON has no space after colons or commas
    assert ": " not in line
    assert ", " not in line
