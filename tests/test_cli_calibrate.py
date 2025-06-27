"""CLI tests for the calibrate subcommand."""

from __future__ import annotations

import json

import pytest

from hypofuse.cli import main


def test_calibrate_basic(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["calibrate", "--scores", "0.1,0.2,0.3"])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data) == 3
    # temperature_scale output sums to ~1.0
    assert abs(sum(data) - 1.0) < 1e-6


def test_calibrate_temperature(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["calibrate", "--scores", "1.0,2.0,3.0", "--temperature", "2.0"])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data) == 3


def test_calibrate_empty_scores(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["calibrate", "--scores", ""])
    assert code == 2
    assert "empty" in capsys.readouterr().err


def test_calibrate_non_numeric(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["calibrate", "--scores", "a,b,c"])
    assert code == 2
    assert "non-numeric" in capsys.readouterr().err
