"""Hardened fusion_invariants: length, range, and monotone agreement checks."""

from __future__ import annotations

from hypofuse.fusion import FusionResult, fusion_invariants
from hypofuse.multi_align import progressive_align


def _make_valid_result():
    grid = progressive_align([("a", "b"), ("a", "c")])
    from hypofuse.fusion import fuse

    return fuse(grid), [("a", "b"), ("a", "c")]


def test_valid_result_passes_all_checks() -> None:
    result, inputs = _make_valid_result()
    assert fusion_invariants(result, inputs) is True


def test_corrupted_length_fails() -> None:
    """confidences shorter than tokens fails."""
    result = FusionResult(
        tokens=("a", "b"),
        confidences=(1.0,),
        chosen=(("a", 1.0), ("b", 1.0)),
    )
    assert fusion_invariants(result, [("a", "b")]) is False


def test_corrupted_confidence_out_of_range() -> None:
    """Confidence > 1.0 fails."""
    result = FusionResult(
        tokens=("a",),
        confidences=(1.5,),
        chosen=(("a", 1.0),),
    )
    assert fusion_invariants(result, [("a",)]) is False


def test_negative_confidence_fails() -> None:
    result = FusionResult(
        tokens=("a",),
        confidences=(-0.1,),
        chosen=(("a", 1.0),),
    )
    assert fusion_invariants(result, [("a",)]) is False


def test_invented_token_fails() -> None:
    """Token not in inputs fails subset check."""
    result = FusionResult(
        tokens=("z",),
        confidences=(1.0,),
        chosen=(("z", 1.0),),
    )
    assert fusion_invariants(result, [("a", "b")]) is False


def test_agreement_above_one_fails() -> None:
    """agreements > 1.0 fails the monotone check."""
    result = FusionResult(
        tokens=("a",),
        confidences=(1.0,),
        chosen=(("a", 1.0),),
        agreements=(1.5,),
    )
    assert fusion_invariants(result, [("a",)]) is False


def test_empty_agreements_skips_check() -> None:
    """Empty agreements tuple does not trigger the agreement check."""
    result = FusionResult(
        tokens=("a",),
        confidences=(1.0,),
        chosen=(("a", 1.0),),
        agreements=(),
    )
    assert fusion_invariants(result, [("a",)]) is True
