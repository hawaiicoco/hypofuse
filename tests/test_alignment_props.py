"""Property-style alignment invariants."""

from __future__ import annotations

from hypofuse.alignment import DEL, INS, MATCH, SUB, edit_alignment


def test_path_walks_every_ref_position() -> None:
    ref = list("abcde")
    hyp = list("abxde")
    align = edit_alignment(ref, hyp)
    ref_steps = [op for op in align.ops if op.op in {MATCH, SUB, DEL}]
    assert len(ref_steps) == len(ref)


def test_path_walks_every_hyp_position() -> None:
    ref = list("abcde")
    hyp = list("abxyde")
    align = edit_alignment(ref, hyp)
    hyp_steps = [op for op in align.ops if op.op in {MATCH, SUB, INS}]
    assert len(hyp_steps) == len(hyp)


def test_score_lower_bound() -> None:
    align = edit_alignment(list("abc"), list("abd"))
    # 1 substitution, no insertions/deletions
    assert align.score >= 1
    assert align.score == 1


def test_score_upper_bound() -> None:
    align = edit_alignment(list("abc"), list("xyz"))
    # At most 3 substitutions
    assert align.score <= 3


def test_errors_bounded_by_max_length() -> None:
    align = edit_alignment(list("abcd"), list("wxyz"))
    assert align.errors <= max(len("abcd"), len("wxyz"))


def test_reorder_breaks_match_count() -> None:
    ref = ["a", "b", "c", "d"]
    aligned = edit_alignment(ref, ref)
    permuted = edit_alignment(ref, list(reversed(ref)))
    assert aligned.errors == 0
    assert permuted.errors > 0
