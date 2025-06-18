"""CLI tests for the align subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hypofuse.cli import main
from hypofuse.fixtures import FixtureConfig, as_manifest_dicts, generate_fixture
from hypofuse.manifests.jsonl import write_manifest


def _write_nbest(tmp_path: Path) -> Path:
    fx = generate_fixture(FixtureConfig(n_utterances=2, n_best=3, seed=1))
    rows = [r for r in as_manifest_dicts(fx) if r["schema"] == "hypofuse.nbest"]
    path = tmp_path / "nbest.jsonl"
    write_manifest(path, rows)
    return path


def test_align_basic(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_nbest(tmp_path)
    code = main(["align", "--nbest", str(path)])
    assert code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert "utterance_id" in obj
        assert obj["depth"] == 3
        assert obj["width"] >= 1


def test_align_missing_file(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["align", "--nbest", "/no/such/file.jsonl"])
    assert code == 2
    assert "hypofuse align" in capsys.readouterr().err
