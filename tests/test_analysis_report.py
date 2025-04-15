"""Markdown and JSONL report export."""

from __future__ import annotations

import json

from hypofuse.analysis import (
    ComparisonRow,
    SliceMetric,
    report_to_jsonl,
    report_to_markdown,
)


def test_markdown_contains_headers() -> None:
    md = report_to_markdown([SliceMetric("all", 5, 0.05, 0.1)], [])
    assert "# Error Analysis Report" in md
    assert "## Slices" in md


def test_markdown_escapes_html_in_slice_keys() -> None:
    md = report_to_markdown([SliceMetric("<script>", 1, 0.0, 0.0)], [])
    assert "<script>" not in md.split("## System comparisons")[0]


def test_jsonl_one_record_per_slice() -> None:
    payload = report_to_jsonl(
        [SliceMetric("a", 1, 0.0, 0.0), SliceMetric("b", 2, 0.0, 0.0)],
        [],
    )
    records = [json.loads(line) for line in payload.splitlines()]
    assert len(records) == 2
    assert all(r["kind"] == "slice" for r in records)


def test_jsonl_one_record_per_comparison() -> None:
    payload = report_to_jsonl(
        [],
        [ComparisonRow("A", "B", 0.01, 0.02, -0.05, 0.07, -0.06, 0.08, 10)],
    )
    records = [json.loads(line) for line in payload.splitlines()]
    assert len(records) == 1
    assert records[0]["kind"] == "comparison"
