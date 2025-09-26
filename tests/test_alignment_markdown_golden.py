"""Frozen golden for alignment_to_markdown output."""

from __future__ import annotations

from hypofuse.alignment import alignment_to_markdown, edit_alignment


def test_markdown_golden_exact() -> None:
    align = edit_alignment(["a", "b", "c"], ["a", "x", "c"])
    md = alignment_to_markdown(align, max_rows=2)
    expected = (
        "| op | ref | hyp |\n"
        "|---|---|---|\n"
        "| MATCH | 'a' | 'a' |\n"
        "| SUB | 'b' | 'x' |\n"
        "| ... (1 more) ... | | |"
    )
    assert md == expected


def test_markdown_no_truncation_when_under_limit() -> None:
    align = edit_alignment(["a", "b"], ["a", "b"])
    md = alignment_to_markdown(align, max_rows=10)
    assert "more" not in md
    lines = md.split("\n")
    # header + separator + 2 data rows = 4 lines
    assert len(lines) == 4


def test_markdown_empty_alignment() -> None:
    align = edit_alignment([], [])
    md = alignment_to_markdown(align)
    lines = md.split("\n")
    # header + separator only
    assert len(lines) == 2
