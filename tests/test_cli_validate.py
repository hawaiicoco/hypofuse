"""CLI tests for the validate subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hypofuse.cli import main
from hypofuse.manifests.jsonl import write_manifest


def _write_manifest(tmp_path: Path, name: str = "m.jsonl") -> Path:
    path = tmp_path / name
    rows = [
        {"schema": "hypofuse.reference", "utterance_id": "u0001", "text": "hello world"},
        {"schema": "hypofuse.reference", "utterance_id": "u0002", "text": "goodbye world"},
    ]
    write_manifest(path, rows)
    return path


def test_validate_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_manifest(tmp_path)
    code = main(["validate", str(path)])
    assert code == 0
    assert "validated 2 records" in capsys.readouterr().out


def test_validate_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_manifest(tmp_path)
    code = main(["validate", str(path), "--json"])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["valid"] is True
    assert data["count"] == 2


def test_validate_nonexistent(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["validate", "/no/such/file.jsonl"])
    assert code == 2
    err = capsys.readouterr().err
    assert "hypofuse validate" in err
    assert "not found" in err


def test_validate_duplicate_ids(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "dup.jsonl"
    rows = [
        {"schema": "hypofuse.reference", "utterance_id": "u0001", "text": "a"},
        {"schema": "hypofuse.reference", "utterance_id": "u0001", "text": "b"},
    ]
    write_manifest(path, rows)
    code = main(["validate", str(path)])
    assert code == 2
    assert "hypofuse validate" in capsys.readouterr().err


def test_validate_schema_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_manifest(tmp_path)
    code = main(["validate", str(path), "--schema", "hypofuse.nbest"])
    assert code == 2
    assert "expected" in capsys.readouterr().err
