"""Detailed manifest artifact checks for ex01 outputs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run_ex01(tmp_path: Path) -> Path:
    """Run ex01 into a subdirectory and return the output path."""
    script = ROOT / "examples" / "ex01_validate_score" / "run.py"
    outdir = tmp_path / "out"
    outdir.mkdir()
    result = subprocess.run(
        [sys.executable, str(script), str(outdir)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return outdir


def test_ex01_nbest_row_count(tmp_path: Path) -> None:
    """FixtureConfig(n_utterances=3) produces exactly 3 nbest rows."""
    outdir = _run_ex01(tmp_path)
    lines = (outdir / "nbest.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        row = json.loads(line)
        assert row["schema"] == "hypofuse.nbest"
        assert "hypotheses" in row


def test_ex01_reference_row_count(tmp_path: Path) -> None:
    """FixtureConfig(n_utterances=3) produces exactly 3 reference rows."""
    outdir = _run_ex01(tmp_path)
    lines = (outdir / "reference.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        row = json.loads(line)
        assert row["schema"] == "hypofuse.reference"


def test_ex01_nbest_has_two_hypotheses(tmp_path: Path) -> None:
    """FixtureConfig(n_best=2) gives 2 hypotheses per n-best list."""
    outdir = _run_ex01(tmp_path)
    lines = (outdir / "nbest.jsonl").read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        row = json.loads(line)
        assert len(row["hypotheses"]) == 2
