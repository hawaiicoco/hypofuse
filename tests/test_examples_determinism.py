"""Determinism: running an example twice yields identical outputs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.slow
def test_ex01_determinism(tmp_path: Path) -> None:
    """Two runs of ex01 produce byte-identical nbest.jsonl."""
    script = ROOT / "examples" / "ex01_validate_score" / "run.py"
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    out_a.mkdir()
    out_b.mkdir()
    for outdir in (out_a, out_b):
        result = subprocess.run(
            [sys.executable, str(script), str(outdir)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    content_a = (out_a / "nbest.jsonl").read_text(encoding="utf-8")
    content_b = (out_b / "nbest.jsonl").read_text(encoding="utf-8")
    assert content_a == content_b, "nbest.jsonl differs between runs"


@pytest.mark.slow
def test_ex02_determinism(tmp_path: Path) -> None:
    """Two runs of ex02 produce byte-identical confusion JSON."""
    script = ROOT / "examples" / "ex02_rover_confusion" / "run.py"
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    out_a.mkdir()
    out_b.mkdir()
    for outdir in (out_a, out_b):
        result = subprocess.run(
            [sys.executable, str(script), str(outdir)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    files_a = sorted(f.name for f in out_a.glob("*_cn.json"))
    files_b = sorted(f.name for f in out_b.glob("*_cn.json"))
    assert files_a == files_b
    for name in files_a:
        text_a = (out_a / name).read_text(encoding="utf-8")
        text_b = (out_b / name).read_text(encoding="utf-8")
        assert text_a == text_b, f"{name} differs between runs"
