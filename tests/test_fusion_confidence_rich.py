"""Tests for richer fused confidence fields: agreements, margins, entropies."""

from __future__ import annotations

import math

import pytest

from hypofuse.fusion import FusionConfig, FusionResult, fuse
from hypofuse.multi_align import GAP, TokenGrid, progressive_align


def test_two_one_split_hand_computed() -> None:
    """3 hypotheses, 2/1 split: agreement=2/3, margin=1/3, entropy=-(2/3*ln(2/3)+1/3*ln(1/3))."""
    grid = progressive_align([("a",), ("b",), ("b",)])
    result = fuse(grid, config=FusionConfig(policy="majority"))
    # 2 vote for "b", 1 for "a" => winner mass=2, runner_up=1, total=3
    assert result.agreements[0] == pytest.approx(2 / 3)
    assert result.margins[0] == pytest.approx(1 / 3)
    expected_entropy = -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3))
    assert result.entropies[0] == pytest.approx(expected_entropy)


def test_unanimous_column() -> None:
    """All hypotheses agree: agreement=1.0, margin=1.0, entropy=0.0."""
    grid = progressive_align([("a",), ("a",), ("a",)])
    result = fuse(grid, config=FusionConfig(policy="majority"))
    assert result.agreements[0] == pytest.approx(1.0)
    assert result.margins[0] == pytest.approx(1.0)
    assert result.entropies[0] == pytest.approx(0.0)


def test_all_gap_column_degenerate() -> None:
    """All-gap column: agreement=0, margin=0, entropy=0."""
    grid = TokenGrid(
        columns=((GAP, GAP),),
        hypotheses=(("a",), ("b",)),
        back_pointers=(((0, "DEL"),), ((0, "DEL"),)),
        pair_alignments=(),
    )
    result = fuse(grid, config=FusionConfig(policy="majority"))
    assert result.agreements[0] == pytest.approx(0.0)
    assert result.margins[0] == pytest.approx(0.0)
    assert result.entropies[0] == pytest.approx(0.0)


def test_confidences_unchanged() -> None:
    """The existing confidences field is unchanged (count-based agreement)."""
    grid = progressive_align([("a", "b"), ("a", "c"), ("x", "y")])
    result = fuse(grid)
    assert result.confidences[0] == pytest.approx(2 / 3)
    # Column 1 holds ("b", "c", "y"): three different tokens, so 1/3 agree.
    assert result.confidences[1] == pytest.approx(1 / 3)


def test_three_way_split() -> None:
    """3 hypotheses all different: agreement=1/3, margin=0, entropy=ln(3)."""
    grid = progressive_align([("a",), ("b",), ("c",)])
    result = fuse(grid, config=FusionConfig(policy="majority"))
    assert result.agreements[0] == pytest.approx(1 / 3)
    assert result.margins[0] == pytest.approx(0.0)
    assert result.entropies[0] == pytest.approx(math.log(3))


def test_result_has_default_empty_tuples() -> None:
    """FusionResult can be constructed without the new fields."""
    r = FusionResult(tokens=("a",), confidences=(1.0,), chosen=(("a", 1.0),))
    assert r.agreements == ()
    assert r.margins == ()
    assert r.entropies == ()


def test_multi_column_lengths_match() -> None:
    """agreements, margins, entropies have the same length as tokens."""
    grid = progressive_align([("a", "b", "c"), ("a", "c"), ("x", "b", "c")])
    result = fuse(grid)
    assert len(result.agreements) == len(result.tokens)
    assert len(result.margins) == len(result.tokens)
    assert len(result.entropies) == len(result.tokens)
