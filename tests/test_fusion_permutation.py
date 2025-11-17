"""Permutation policy tests: order_policy given vs sorted."""

from __future__ import annotations

import pytest

from hypofuse.exceptions import FusionError
from hypofuse.fusion import FusionConfig, fuse
from hypofuse.multi_align import progressive_align


def test_given_first_depends_on_hypothesis_order() -> None:
    """Under order_policy='given' + tie_break='first', permuting hyps can change output."""
    grid_a = progressive_align([("x",), ("y",)])
    grid_b = progressive_align([("y",), ("x",)])
    cfg = FusionConfig(tie_break="first", order_policy="given")
    r_a = fuse(grid_a, config=cfg)
    r_b = fuse(grid_b, config=cfg)
    # "first" picks the first candidate, which depends on hypothesis order.
    assert r_a.tokens[0] != r_b.tokens[0]


def test_sorted_invariant_to_hypothesis_order() -> None:
    """Under order_policy='sorted', permuting hypotheses does not change output."""
    grid_a = progressive_align([("x",), ("y",)])
    grid_b = progressive_align([("y",), ("x",)])
    cfg = FusionConfig(tie_break="first", order_policy="sorted")
    r_a = fuse(grid_a, config=cfg)
    r_b = fuse(grid_b, config=cfg)
    assert r_a.tokens == r_b.tokens


def test_sorted_with_lexicographic_matches_given() -> None:
    """sorted + lexicographic gives the same result as given + lexicographic."""
    grid = progressive_align([("b",), ("a",)])
    cfg_given = FusionConfig(tie_break="lexicographic", order_policy="given")
    cfg_sorted = FusionConfig(tie_break="lexicographic", order_policy="sorted")
    assert fuse(grid, config=cfg_given).tokens == fuse(grid, config=cfg_sorted).tokens


def test_unknown_order_policy_raises() -> None:
    with pytest.raises(FusionError, match="order_policy"):
        FusionConfig(order_policy="random").validate()


def test_given_default_is_backward_compatible() -> None:
    """Default order_policy='given' preserves existing behaviour."""
    grid = progressive_align([("a", "b"), ("a", "c"), ("x", "y")])
    result = fuse(grid)
    assert result.tokens[0] == "a"


def test_sorted_multi_column() -> None:
    """sorted is applied per column independently."""
    grid = progressive_align([("b", "d"), ("a", "c")])
    cfg = FusionConfig(tie_break="first", order_policy="sorted")
    result = fuse(grid, config=cfg)
    # After sorting candidates by token: col0 has ("a","b"), col1 has ("c","d")
    # "first" picks the first in sorted order: "a" and "c"
    assert result.tokens == ("a", "c")
