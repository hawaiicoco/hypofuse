"""CLI tests for the report subcommand."""

from __future__ import annotations

from pathlib import Path

import pytest

from hypofuse.cli import main
from hypofuse.fixtures import FixtureConfig, as_manifest_dicts, generate_fixture
from hypofuse.manifests.jsonl import write_manifest


def _write_pair(tmp_path: Path) -> tuple[Path, Path]:
    fx = generate_fixture(FixtureConfig(n_utterances=4, n_best=2, seed=5))
    rows = as_manifest_dicts(fx)
    nb = [r for r in rows if r["schema"] == "hypofuse.nbest"]
    ref = [r for r in rows if r["schema"] == "hypofuse.reference"]
    nb_path = tmp_path / "nbest.jsonl"
    ref_path = tmp_path / "ref.jsonl"
    write_manifest(nb_path, nb)
    write_manifest(ref_path, ref)
    return nb_path, ref_path


def test_report_markdown_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    nb, ref = _write_pair(tmp_path)
    code = main(["report", "--nbest", str(nb), "--reference", str(ref), "--format", "markdown"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Error Analysis Report" in out or "Slice" in out


def test_report_jsonl_to_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    nb, ref = _write_pair(tmp_path)
    out_path = tmp_path / "report.jsonl"
    code = main(
        [
            "report",
            "--nbest",
            str(nb),
            "--reference",
            str(ref),
            "--format",
            "jsonl",
            "--out",
            str(out_path),
        ]
    )
    assert code == 0
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "slice" in content


def test_report_missing_file(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["report", "--nbest", "/no.jsonl", "--reference", "/x.jsonl"])
    assert code == 2
    assert "hypofuse report" in capsys.readouterr().err
