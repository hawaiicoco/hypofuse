"""CLI tests for the demo subcommand."""

from __future__ import annotations

from pathlib import Path

import pytest

from hypofuse.cli import main


def test_demo_default(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["demo", "--out", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "nbest.jsonl").exists()
    assert (tmp_path / "reference.jsonl").exists()
    assert (tmp_path / "report.md").exists()
    out = capsys.readouterr().out
    assert "CER=" in out
    assert "WER=" in out


def test_demo_custom_params(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "demo",
            "--out",
            str(tmp_path),
            "--utterances",
            "3",
            "--n-best",
            "2",
            "--seed",
            "42",
        ]
    )
    assert code == 0
    assert (tmp_path / "report.md").exists()
    content = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Utterances: 3" in content


def test_demo_creates_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    nested = tmp_path / "a" / "b" / "c"
    code = main(["demo", "--out", str(nested)])
    assert code == 0
    assert nested.exists()
    assert (nested / "nbest.jsonl").exists()
