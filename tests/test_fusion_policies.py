"""Policy validation and edge cases."""

from __future__ import annotations

import pytest

from hypofuse.exceptions import FusionError
from hypofuse.fusion import FusionConfig, fuse
from hypofuse.multi_align import progressive_align


def test_unknown_policy_raises() -> None:
    with pytest.raises(FusionError):
        FusionConfig(policy="bogus").validate()


def test_negative_alpha_raises() -> None:
    with pytest.raises(FusionError):
        FusionConfig(alpha=-0.1).validate()


def test_unknown_tie_break_raises() -> None:
    with pytest.raises(FusionError):
        FusionConfig(tie_break="random").validate()


def test_fuse_rejects_unknown_policy() -> None:
    grid = progressive_align([("a",), ("b",)])
    with pytest.raises(FusionError):
        fuse(grid, config=FusionConfig(policy="bogus"))


def test_fuse_with_all_gap_columns() -> None:
    grid = progressive_align([("a", "b"), ("c", "d")])
    # Force a configuration where a column has only gap tokens.
    empty_col_grid = type(grid)(
        columns=(("*",),),
        hypotheses=(("a",), ("b",)),
        back_pointers=grid.back_pointers,
        pair_alignments=grid.pair_alignments,
    )
    result = fuse(empty_col_grid)
    assert result.tokens[0] == "*"


def test_fuse_missing_scores_default_zero() -> None:
    grid = progressive_align([("a",), ("b",)])
    result = fuse(grid)
    # No scores -> uniform weights; tie goes lexicographically to "a".
    assert result.tokens[0] == "a"
