"""Tests for report_from_row / report_to_row."""

from __future__ import annotations

import json

import pytest

from hypofuse.convert import report_from_row, report_to_row
from hypofuse.exceptions import SchemaError
from hypofuse.manifests.reporter import ReportRecord


def _row(**overrides: object) -> dict:
    base: dict = {
        "schema": "hypofuse.report",
        "report_id": "r1",
        "generated_at": "2026-01-01T00:00:00Z",
        "hypofuse_version": "0.1.0",
        "config_hash": "abc",
        "seed": 42,
    }
    base.update(overrides)
    return base


def test_report_roundtrip() -> None:
    original = ReportRecord(
        report_id="r1",
        schema="hypofuse.report",
        generated_at="2026-01-01T00:00:00Z",
        hypofuse_version="0.1.0",
        config_hash="abc",
        seed=42,
        artifact_paths=("report.md",),
        metrics=(("wer", 0.1), ("cer", 0.05)),
    )
    row = report_to_row(original)
    restored = report_from_row(row)
    assert restored == original
    assert json.loads(json.dumps(row)) == row


def test_report_missing_required() -> None:
    row = _row()
    del row["seed"]
    with pytest.raises(SchemaError, match="missing required field: seed"):
        report_from_row(row)


def test_report_wrong_type() -> None:
    with pytest.raises(SchemaError, match="field seed expected int"):
        report_from_row(_row(seed="42"))


def test_report_unknown_key() -> None:
    with pytest.raises(SchemaError, match="unknown keys"):
        report_from_row(_row(bogus=1))
