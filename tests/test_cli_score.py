"""CLI tests for the score subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hypofuse.cli import main
from hypofuse.fixtures import FixtureConfig, as_manifest_dicts, generate_fixture
from hypofuse.manifests.jsonl import write_manifest


def _write_pair(tmp_path: Path) -> tuple[Path, Path]:
    fx = generate_fixture(FixtureConfig(n_utterances=3, n_best=2, seed=7))
    rows = as_manifest_dicts(fx)
    nbest = [r for r in rows if r["schema"] == "hypofuse.nbest"]
    refs = [r for r in rows if r["schema"] == "hypofuse.reference"]
    nb_path = tmp_path / "nbest.jsonl"
    ref_path = tmp_path / "ref.jsonl"
    write_manifest(nb_path, nbest)
    write_manifest(ref_path, refs)
    return nb_path, ref_path


def test_score_both(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    nb, ref = _write_pair(tmp_path)
    code = main(["score", "--nbest", str(nb), "--reference", str(ref)])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert "cer" in data
    assert "wer" in data
    assert data["n"] == 3


def test_score_cer_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    nb, ref = _write_pair(tmp_path)
    code = main(["score", "--nbest", str(nb), "--reference", str(ref), "--metric", "cer"])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert "cer" in data
    assert "wer" not in data


def test_score_no_match(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    nb_path = tmp_path / "nb.jsonl"
    ref_path = tmp_path / "ref.jsonl"
    write_manifest(
        nb_path,
        [
            {
                "schema": "hypofuse.nbest",
                "utterance_id": "x1",
                "system": "s",
                "language": "en",
                "hypotheses": [{"rank": 1, "text": "hi", "tokens": ["hi"]}],
            }
        ],
    )
    write_manifest(
        ref_path,
        [{"schema": "hypofuse.reference", "utterance_id": "y1", "text": "bye"}],
    )
    code = main(["score", "--nbest", str(nb_path), "--reference", str(ref_path)])
    assert code == 2
    assert "no matched" in capsys.readouterr().err
