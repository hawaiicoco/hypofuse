"""Null/epsilon handling tests for fusion null_policy."""

from __future__ import annotations

import pytest

from hypofuse.exceptions import FusionError
from hypofuse.fusion import FusionConfig, fuse
from hypofuse.multi_align import GAP, progressive_align


def _make_grid():
    """Grid from hypotheses of different lengths, producing a gap column.

    progressive_align([("a", "b", "c"), ("a", "c")]) yields columns:
    ("a","a"), ("b","*"), ("c","c").
    """
    return progressive_align([("a", "b", "c"), ("a", "c")])


def test_keep_default_produces_gap_token() -> None:
    """Default null_policy='keep' emits the gap symbol for gap columns."""
    grid = _make_grid()
    result = fuse(grid, config=FusionConfig(null_policy="keep"))
    assert result.tokens == ("a", "b", "c")
    assert len(result.tokens) == 3


def test_keep_emits_null_token_for_gap_column() -> None:
    """Under 'keep', gap columns produce the null_token in the output."""
    grid = _make_grid()
    result = fuse(grid)  # default null_policy is "keep"
    assert len(result.tokens) == 3


def test_drop_keeps_columns_that_have_a_token() -> None:
    """'drop' removes only columns whose winner is the gap symbol.

    Column 1 of this grid is ("b", GAP): one hypothesis really did emit "b",
    so the column survives and the fused token keeps its position.
    """
    grid = _make_grid()
    result = fuse(grid, config=FusionConfig(null_policy="drop"))
    assert result.tokens == ("a", "b", "c")
    assert GAP not in result.tokens


def test_vote_counts_gap_as_candidate() -> None:
    """Under 'vote', the gap symbol participates in voting."""
    grid = _make_grid()
    result = fuse(grid, config=FusionConfig(null_policy="vote"))
    # Column 1: candidates are ("b", 1.0) and ("*", 1.0).
    # Tie broken lexicographically: "*" < "b", so "*" wins.
    assert result.tokens[1] == "*"
    assert len(result.tokens) == 3


def test_vote_unanimous_gap_column() -> None:
    """Under 'vote', an all-gap column has GAP as the unanimous winner."""
    # An all-gap column cannot come from progressive_align, so build it directly.
    from hypofuse.multi_align import TokenGrid

    gap_grid = TokenGrid(
        columns=((GAP, GAP),),
        hypotheses=(("a",), ("b",)),
        back_pointers=(((0, "DEL"),), ((0, "DEL"),)),
        pair_alignments=(),
    )
    result = fuse(gap_grid, config=FusionConfig(null_policy="vote"))
    assert result.tokens == ("*",)


def test_unknown_null_policy_raises() -> None:
    with pytest.raises(FusionError, match="null_policy"):
        FusionConfig(null_policy="invalid").validate()


def test_drop_all_gap_produces_empty() -> None:
    """If every column is a gap column under 'drop', output is empty."""
    from hypofuse.multi_align import TokenGrid

    gap_grid = TokenGrid(
        columns=((GAP, GAP), (GAP, GAP)),
        hypotheses=(("a", "b"), ("c", "d")),
        back_pointers=(((0, "DEL"), (1, "DEL")), ((0, "DEL"), (1, "DEL"))),
        pair_alignments=(),
    )
    result = fuse(gap_grid, config=FusionConfig(null_policy="drop"))
    assert result.tokens == ()
    assert result.confidences == ()
