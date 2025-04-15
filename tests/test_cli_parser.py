"""CLI argument parser shape.

The parser is public surface: every documented subcommand must be reachable
and must render usable help text.
"""

from __future__ import annotations

import argparse

import pytest

from hypofuse.cli import _build_parser

COMMANDS = (
    "validate",
    "normalize",
    "score",
    "align",
    "fuse",
    "rescore",
    "calibrate",
    "analyze",
    "report",
    "demo",
)


def _subparser_choices(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    raise AssertionError("hypofuse parser exposes no subcommands")


def test_all_subcommands_registered() -> None:
    choices = _subparser_choices(_build_parser())
    for cmd in COMMANDS:
        assert cmd in choices, cmd


def test_no_undocumented_subcommands() -> None:
    assert set(_subparser_choices(_build_parser())) == set(COMMANDS)


def test_help_lists_every_subcommand_with_text() -> None:
    text = _build_parser().format_help()
    for cmd in COMMANDS:
        assert cmd in text, cmd
    # Each registered command contributes a help line beyond the usage block.
    assert len([line for line in text.splitlines() if line.strip()]) > len(COMMANDS)


def test_subcommand_help_exits_zero_and_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    parser = _build_parser()
    for cmd in ("fuse", "demo", "calibrate"):
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args([cmd, "--help"])
        assert excinfo.value.code == 0
        assert capsys.readouterr().out.startswith("usage: hypofuse")


def test_missing_command_is_a_parser_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _build_parser().parse_args([])
    assert excinfo.value.code == 2


def test_unknown_command_is_rejected() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _build_parser().parse_args(["transmogrify"])
    assert excinfo.value.code == 2


def test_version_flag_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    from hypofuse import __version__

    with pytest.raises(SystemExit) as excinfo:
        _build_parser().parse_args(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out
