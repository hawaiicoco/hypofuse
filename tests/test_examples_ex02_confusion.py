"""Confusion network output assertions for ex02."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from hypofuse.confusion import confusion_from_json

ROOT = Path(__file__).resolve().parent.parent


def test_ex02_confusion_json_valid(tmp_path: Path) -> None:
    """Each confusion network JSON file parses and validates."""
    script = ROOT / "examples" / "ex02_rover_confusion" / "run.py"
    outdir = tmp_path / "out"
    outdir.mkdir()
    result = subprocess.run(
        [sys.executable, str(script), str(outdir)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    cn_files = sorted(outdir.glob("*_cn.json"))
    # FixtureConfig(n_utterances=4) produces 4 utterances.
    assert len(cn_files) == 4
    for cn_path in cn_files:
        cn = confusion_from_json(cn_path.read_text(encoding="utf-8"))
        cn.validate()
        assert len(cn.slots) > 0


def test_ex02_one_best_matches_pivot(tmp_path: Path) -> None:
    """The one-best path matches the pivot of each slot."""
    script = ROOT / "examples" / "ex02_rover_confusion" / "run.py"
    outdir = tmp_path / "out"
    outdir.mkdir()
    result = subprocess.run(
        [sys.executable, str(script), str(outdir)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for cn_path in sorted(outdir.glob("*_cn.json")):
        cn = confusion_from_json(cn_path.read_text(encoding="utf-8"))
        one_best = cn.one_best()
        pivots = tuple(slot.pivot for slot in cn.slots)
        assert one_best == pivots
