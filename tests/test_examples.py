"""Subprocess invocation tests for each example and the runner.

Each test runs an example via subprocess with ``sys.executable`` into
a temporary directory, asserting exit code 0 and that expected
artifacts exist and are non-empty.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"
SCRIPTS_DIR = ROOT / "scripts"


def _run_example(name: str, tmpdir: Path) -> subprocess.CompletedProcess[str]:
    """Invoke an example script into *tmpdir* / name."""
    script = EXAMPLES_DIR / name / "run.py"
    outdir = tmpdir / name
    outdir.mkdir()
    return subprocess.run(
        [sys.executable, str(script), str(outdir)],
        capture_output=True,
        text=True,
    )


def test_ex01_validate_score(tmp_path: Path) -> None:
    """ex01 exits 0 and writes non-empty manifests."""
    result = _run_example("ex01_validate_score", tmp_path)
    assert result.returncode == 0, result.stderr
    outdir = tmp_path / "ex01_validate_score"
    assert (outdir / "nbest.jsonl").stat().st_size > 0
    assert (outdir / "reference.jsonl").stat().st_size > 0


def test_ex02_rover_confusion(tmp_path: Path) -> None:
    """ex02 exits 0 and writes confusion network JSON files."""
    result = _run_example("ex02_rover_confusion", tmp_path)
    assert result.returncode == 0, result.stderr
    cn_files = list((tmp_path / "ex02_rover_confusion").glob("*_cn.json"))
    assert len(cn_files) > 0
    for cn_path in cn_files:
        assert cn_path.stat().st_size > 0


def test_ex03_rescore_calibrate(tmp_path: Path) -> None:
    """ex03 exits 0 and writes a non-empty Markdown report."""
    result = _run_example("ex03_rescore_calibrate", tmp_path)
    assert result.returncode == 0, result.stderr
    report = tmp_path / "ex03_rescore_calibrate" / "report.md"
    assert report.stat().st_size > 0


@pytest.mark.slow
def test_run_examples_script(tmp_path: Path) -> None:
    """scripts/run_examples.py exits 0."""
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "run_examples.py"), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
