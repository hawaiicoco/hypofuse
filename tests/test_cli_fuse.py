"""CLI tests for the fuse subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hypofuse.cli import main
from hypofuse.fixtures import FixtureConfig, as_manifest_dicts, generate_fixture
from hypofuse.manifests.jsonl import write_manifest


def _write_nbest(tmp_path: Path) -> Path:
    fx = generate_fixture(FixtureConfig(n_utterances=2, n_best=3, seed=2))
    rows = [r for r in as_manifest_dicts(fx) if r["schema"] == "hypofuse.nbest"]
    path = tmp_path / "nbest.jsonl"
    write_manifest(path, rows)
    return path


def test_fuse_majority(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_nbest(tmp_path)
    code = main(["fuse", "--nbest", str(path), "--policy", "majority"])
    assert code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) >= 1
    for line in lines:
        obj = json.loads(line)
        assert "tokens" in obj
        assert "confidences" in obj


def test_fuse_score_weighted(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_nbest(tmp_path)
    code = main(["fuse", "--nbest", str(path), "--policy", "score_weighted"])
    assert code == 0
    assert capsys.readouterr().out.strip()


def test_fuse_missing_file(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["fuse", "--nbest", "/no/such/file.jsonl"])
    assert code == 2
    assert "hypofuse fuse" in capsys.readouterr().err
