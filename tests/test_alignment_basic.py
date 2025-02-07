"""Golden alignment cases."""

from __future__ import annotations

from hypofuse.alignment import (
    DEL,
    INS,
    MATCH,
    SUB,
    edit_alignment,
)


def test_alignment_perfect_match() -> None:
    align = edit_alignment(["a", "b", "c"], ["a", "b", "c"])
    assert all(op.op == MATCH for op in align.ops)
    assert align.score == 0
    assert align.errors == 0


def test_alignment_single_substitution() -> None:
    align = edit_alignment(["a", "b", "c"], ["a", "x", "c"])
    assert [op.op for op in align.ops] == [MATCH, SUB, MATCH]
    assert align.errors == 1


def test_alignment_single_insertion() -> None:
    align = edit_alignment(["a", "b"], ["a", "x", "b"])
    assert [op.op for op in align.ops] == [MATCH, INS, MATCH]


def test_alignment_single_deletion() -> None:
    align = edit_alignment(["a", "b", "c"], ["a", "c"])
    assert [op.op for op in align.ops] == [MATCH, DEL, MATCH]


def test_alignment_empty_reference() -> None:
    align = edit_alignment([], ["a"])
    assert [op.op for op in align.ops] == [INS]
    assert align.errors == 1


def test_alignment_empty_hypothesis() -> None:
    align = edit_alignment(["a"], [])
    assert [op.op for op in align.ops] == [DEL]
    assert align.errors == 1


def test_alignment_both_empty() -> None:
    align = edit_alignment([], [])
    assert align.ops == ()


def test_alignment_score_matches_errors() -> None:
    align = edit_alignment(["a", "b", "c"], ["a", "x", "y", "c"])
    assert align.score == align.errors


def test_alignment_ref_hyp_lengths() -> None:
    align = edit_alignment(["a", "b", "c"], ["a", "x", "c", "z"])
    assert align.ref_length == 3
    assert align.hyp_length == 4


def test_alignment_op_carries_tokens() -> None:
    align = edit_alignment(["a", "b"], ["a", "x"])
    sub = next(op for op in align.ops if op.op == SUB)
    assert sub.ref_token == "b"
    assert sub.hyp_token == "x"


def test_alignment_rejects_string_input() -> None:
    import pytest

    from hypofuse.exceptions import AlignmentError

    with pytest.raises(AlignmentError):
        edit_alignment("abc", "abd")
