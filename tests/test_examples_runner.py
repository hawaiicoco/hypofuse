"""Runner-specific checks for scripts/run_examples.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_runner_creates_subdirs(tmp_path: Path) -> None:
    """The runner creates a subdirectory for each example."""
    script = ROOT / "scripts" / "run_examples.py"
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    expected = (
        "ex01_validate_score",
        "ex02_rover_confusion",
        "ex03_rescore_calibrate",
    )
    for name in expected:
        assert (tmp_path / name).is_dir(), f"missing directory: {name}"


def test_runner_output_has_exit_codes(tmp_path: Path) -> None:
    """The runner prints one status line per example."""
    script = ROOT / "scripts" / "run_examples.py"
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) >= 3
    for line in lines:
        assert "exit=" in line
