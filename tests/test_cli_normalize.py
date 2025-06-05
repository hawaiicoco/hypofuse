"""CLI tests for the normalize subcommand."""

from __future__ import annotations

import json

import pytest

from hypofuse.cli import main


def test_normalize_default_lowercases(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["normalize", "--reference", "Hello World", "--hypothesis", "HELLO WORLD"])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["reference"] == "hello world"
    assert data["hypothesis"] == "hello world"


def test_normalize_keep_case(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "normalize",
            "--reference",
            "Hello World",
            "--hypothesis",
            "Hello World",
            "--keep-case",
        ]
    )
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["reference"] == "Hello World"


def test_normalize_language_zh(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "normalize",
            "--reference",
            "hello world",
            "--hypothesis",
            "hello world",
            "--language",
            "zh",
        ]
    )
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert "reference" in data
    assert "hypothesis" in data
