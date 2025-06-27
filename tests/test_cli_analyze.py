"""CLI tests for the analyze subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hypofuse.cli import main
from hypofuse.fixtures import FixtureConfig, as_manifest_dicts, generate_fixture
from hypofuse.manifests.jsonl import write_manifest


def _write_pair(tmp_path: Path) -> tuple[Path, Path]:
    fx = generate_fixture(FixtureConfig(n_utterances=5, n_best=2, seed=4))
    rows = as_manifest_dicts(fx)
    nb = [r for r in rows if r["schema"] == "hypofuse.nbest"]
    ref = [r for r in rows if r["schema"] == "hypofuse.reference"]
    nb_path = tmp_path / "nbest.jsonl"
    ref_path = tmp_path / "ref.jsonl"
    write_manifest(nb_path, nb)
    write_manifest(ref_path, ref)
    return nb_path, ref_path


def test_analyze_speaker_group(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    nb, ref = _write_pair(tmp_path)
    code = main(
        [
            "analyze",
            "--nbest",
            str(nb),
            "--reference",
            str(ref),
            "--slice-by",
            "speaker_group",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Slice" in out
    assert "Count" in out


def test_analyze_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    nb, ref = _write_pair(tmp_path)
    code = main(
        [
            "analyze",
            "--nbest",
            str(nb),
            "--reference",
            str(ref),
            "--slice-by",
            "duration_s",
            "--json",
        ]
    )
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert isinstance(data, list)
    assert all("key" in d for d in data)


def test_analyze_missing_file(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["analyze", "--nbest", "/no.jsonl", "--reference", "/x.jsonl"])
    assert code == 2
    assert "hypofuse analyze" in capsys.readouterr().err
