"""JSONL report rows carry schema name and version."""

from __future__ import annotations

import json

from hypofuse.analysis import (
    ComparisonRow,
    SliceMetric,
    report_to_jsonl,
)


def test_every_row_has_schema_keys() -> None:
    payload = report_to_jsonl(
        [SliceMetric("a", 1, 0.0, 0.0)],
        [ComparisonRow("A", "B", 0.01, 0.02, -0.05, 0.07, -0.06, 0.08, 10)],
    )
    for line in payload.splitlines():
        row = json.loads(line)
        assert row["schema"] == "hypofuse.report"
        assert row["schema_version"] == 1


def test_slice_row_golden() -> None:
    payload = report_to_jsonl([SliceMetric("all", 5, 0.1, 0.2)], [])
    row = json.loads(payload.splitlines()[0])
    assert row == {
        "cer": 0.1,
        "count": 5,
        "key": "all",
        "kind": "slice",
        "schema": "hypofuse.report",
        "schema_version": 1,
        "wer": 0.2,
    }


def test_comparison_row_has_schema() -> None:
    payload = report_to_jsonl(
        [],
        [ComparisonRow("A", "B", 0.01, 0.02, -0.05, 0.07, -0.06, 0.08, 10)],
    )
    row = json.loads(payload.splitlines()[0])
    assert row["schema"] == "hypofuse.report"
    assert row["schema_version"] == 1
    assert row["kind"] == "comparison"


def test_all_rows_parse_as_json() -> None:
    payload = report_to_jsonl(
        [SliceMetric("a", 1, 0.0, 0.0), SliceMetric("b", 2, 0.0, 0.0)],
        [ComparisonRow("A", "B", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5)],
    )
    rows = [json.loads(ln) for ln in payload.splitlines()]
    assert len(rows) == 3
