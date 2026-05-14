"""A closed downstream pipe must not produce a traceback or a failure code.

Piping any subcommand into ``head`` closes stdout early. The CLI reports
success in that case: the reader got what it asked for.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from hypofuse.cli import main
from hypofuse.fixtures import FixtureConfig, as_manifest_dicts, generate_fixture
from hypofuse.manifests.jsonl import write_manifest


class _ClosedPipe:
    """Stand-in for a stdout whose reader has gone away."""

    def write(self, *_args: Any) -> int:
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self) -> None:
        return None


@pytest.fixture()
def nbest_path(tmp_path: Path) -> Path:
    fixtures = generate_fixture(FixtureConfig(n_utterances=3, n_best=2, seed=4))
    rows = [r for r in as_manifest_dicts(fixtures) if r["schema"] == "hypofuse.nbest"]
    path = tmp_path / "nbest.jsonl"
    write_manifest(path, rows)
    return path


def test_closed_stdout_returns_zero(monkeypatch: pytest.MonkeyPatch, nbest_path: Path) -> None:
    monkeypatch.setattr(sys, "stdout", _ClosedPipe())
    assert main(["align", "--nbest", str(nbest_path)]) == 0


def test_closed_stdout_on_fuse_returns_zero(
    monkeypatch: pytest.MonkeyPatch, nbest_path: Path
) -> None:
    monkeypatch.setattr(sys, "stdout", _ClosedPipe())
    assert main(["fuse", "--nbest", str(nbest_path)]) == 0


def test_user_error_still_reports_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
    nbest_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The broken-pipe handling must not swallow real errors."""
    monkeypatch.setattr(sys, "stdout", _ClosedPipe())
    missing = nbest_path.with_name("absent.jsonl")
    assert main(["validate", str(missing)]) == 2
    assert "file not found" in capsys.readouterr().err
