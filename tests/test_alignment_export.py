"""Alignment export and table rendering."""

from __future__ import annotations

from hypofuse.alignment import (
    MATCH,
    SUB,
    AlignmentOp,
    alignment_to_markdown,
    edit_alignment,
)


def test_to_dict_round_trip() -> None:
    align = edit_alignment(["a", "b", "c"], ["a", "x", "c"])
    payload = align.to_dict()
    assert payload[0] == {"op": MATCH, "ref": "a", "hyp": "a"}
    assert payload[1] == {"op": SUB, "ref": "b", "hyp": "x"}
    assert payload[2] == {"op": MATCH, "ref": "c", "hyp": "c"}


def test_markdown_render_starts_with_header() -> None:
    align = edit_alignment(["a", "b"], ["a", "b"])
    md = alignment_to_markdown(align)
    assert md.startswith("| op | ref | hyp |")
    assert "| MATCH | 'a' | 'a' |" in md


def test_markdown_render_truncates() -> None:
    ops = tuple(AlignmentOp(MATCH, f"r{i}", f"h{i}") for i in range(50))
    align = type("A", (), {"ops": ops})()  # type: ignore[abstract]
    align.ops = ops
    md = alignment_to_markdown(align, max_rows=5)  # type: ignore[arg-type]
    assert "more" in md
