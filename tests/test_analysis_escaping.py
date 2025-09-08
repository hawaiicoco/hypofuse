"""Markdown escaping in report tables."""

from __future__ import annotations

import re

from hypofuse.analysis import (
    ComparisonRow,
    SliceMetric,
    report_to_markdown,
)


def test_pipe_in_slice_name_escaped() -> None:
    md = report_to_markdown([SliceMetric("a|b", 1, 0.0, 0.0)], [])
    # The pipe must be escaped
    assert "a\\|b" in md

    # Every data row should have the same pipe count as the header
    def separators(line: str) -> int:
        # An escaped pipe (\|) is cell content, not a column separator.
        return len(re.findall(r"(?<!\\)\|", line))

    # Each table is checked against its own header row.
    header: int | None = None
    for line in md.split("\n"):
        if not line.startswith("|"):
            header = None
            continue
        if header is None:
            header = separators(line)
        else:
            assert separators(line) == header, line


def test_backslash_in_slice_name_escaped() -> None:
    md = report_to_markdown([SliceMetric("a\\b", 1, 0.0, 0.0)], [])
    assert "a\\\\b" in md


def test_newline_in_slice_name_escaped() -> None:
    md = report_to_markdown([SliceMetric("a\nb", 1, 0.0, 0.0)], [])
    # The table row must be a single line (no real newline in the cell)
    table_rows = [ln for ln in md.split("\n") if "| a" in ln]
    assert len(table_rows) == 1


def test_system_name_escaped() -> None:
    comp = ComparisonRow("A|B", "C", 0.01, 0.02, -0.05, 0.07, -0.06, 0.08, 10)
    md = report_to_markdown([], [comp])
    assert "A\\|B" in md
