"""AlignmentCosts default behaviour matches unit-cost alignment."""

from __future__ import annotations

import pytest

from hypofuse.alignment import AlignmentCosts, edit_alignment


def test_default_costs_match_no_costs_substitution() -> None:
    ref = ["a", "b", "c"]
    hyp = ["a", "x", "c"]
    a = edit_alignment(ref, hyp)
    b = edit_alignment(ref, hyp, costs=AlignmentCosts())
    assert a.ops == b.ops
    assert a.score == pytest.approx(b.score)


def test_default_costs_match_no_costs_insertion() -> None:
    ref = ["a", "b"]
    hyp = ["a", "x", "b"]
    a = edit_alignment(ref, hyp)
    b = edit_alignment(ref, hyp, costs=AlignmentCosts())
    assert a.ops == b.ops
    assert a.score == pytest.approx(b.score)


def test_default_costs_match_no_costs_deletion() -> None:
    ref = ["a", "b", "c"]
    hyp = ["a", "c"]
    a = edit_alignment(ref, hyp)
    b = edit_alignment(ref, hyp, costs=AlignmentCosts())
    assert a.ops == b.ops
    assert a.score == pytest.approx(b.score)


def test_default_costs_empty() -> None:
    a = edit_alignment([], [])
    b = edit_alignment([], [], costs=AlignmentCosts())
    assert a.ops == b.ops
    assert a.score == pytest.approx(b.score)


def test_validate_rejects_negative_substitution() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        AlignmentCosts(substitution=-1.0).validate()


def test_validate_rejects_nan_insertion() -> None:
    with pytest.raises(ValueError, match="finite"):
        AlignmentCosts(insertion=float("nan")).validate()


def test_validate_rejects_inf_deletion() -> None:
    with pytest.raises(ValueError, match="finite"):
        AlignmentCosts(deletion=float("inf")).validate()


def test_validate_rejects_negative_token_cost() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        AlignmentCosts(token_costs={("a", "b"): -0.5}).validate()


def test_validate_accepts_default() -> None:
    AlignmentCosts().validate()
