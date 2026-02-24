"""Markdown report content checks for ex03."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_ex03_report_has_sections(tmp_path: Path) -> None:
    """The report contains expected Markdown sections."""
    script = ROOT / "examples" / "ex03_rescore_calibrate" / "run.py"
    outdir = tmp_path / "out"
    outdir.mkdir()
    result = subprocess.run(
        [sys.executable, str(script), str(outdir)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    text = (outdir / "report.md").read_text(encoding="utf-8")
    assert "# Error Analysis Report" in text
    assert "## Slices" in text


def test_ex03_report_labels_synthetic(tmp_path: Path) -> None:
    """The report explicitly labels data as synthetic."""
    script = ROOT / "examples" / "ex03_rescore_calibrate" / "run.py"
    outdir = tmp_path / "out"
    outdir.mkdir()
    result = subprocess.run(
        [sys.executable, str(script), str(outdir)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    text = (outdir / "report.md").read_text(encoding="utf-8")
    assert "synthetic" in text.lower()


def test_ex03_report_has_run_meta(tmp_path: Path) -> None:
    """The report includes a Run metadata section."""
    script = ROOT / "examples" / "ex03_rescore_calibrate" / "run.py"
    outdir = tmp_path / "out"
    outdir.mkdir()
    result = subprocess.run(
        [sys.executable, str(script), str(outdir)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    text = (outdir / "report.md").read_text(encoding="utf-8")
    assert "## Run" in text
