"""Posterior-weighted voting policy tests."""

from __future__ import annotations

import pytest

from hypofuse.exceptions import FusionError
from hypofuse.fusion import FusionConfig, fuse
from hypofuse.multi_align import progressive_align


def test_posterior_weighted_flips_winner() -> None:
    """Minority token with high posterior mass beats plain majority."""
    # 3 hypotheses: hyp0="a", hyp1="b", hyp2="b"
    # Plain majority: "b" wins (2 vs 1).
    # Posteriors: hyp0=0.9, hyp1=0.05, hyp2=0.05
    # Weighted: "a"=1*0.9=0.9, "b"=1*0.05+1*0.05=0.1 => "a" wins.
    grid = progressive_align([("a",), ("b",), ("b",)])
    result = fuse(
        grid,
        posteriors=[[0.9], [0.05], [0.05]],
        config=FusionConfig(policy="posterior_weighted"),
    )
    assert result.tokens[0] == "a"


def test_posterior_weighted_majority_agree_without_posteriors() -> None:
    """With uniform posteriors the result matches plain majority."""
    grid = progressive_align([("a",), ("b",), ("b",)])
    result = fuse(
        grid,
        posteriors=[[1.0], [1.0], [1.0]],
        config=FusionConfig(policy="posterior_weighted"),
    )
    assert result.tokens[0] == "b"


def test_posterior_weighted_missing_raises() -> None:
    """posterior_weighted without posteriors raises FusionError."""
    grid = progressive_align([("a",), ("b",)])
    with pytest.raises(FusionError, match="posteriors"):
        fuse(grid, config=FusionConfig(policy="posterior_weighted"))


def test_posterior_wrong_row_count_raises() -> None:
    """Posteriors with wrong number of rows raises FusionError."""
    grid = progressive_align([("a",), ("b",)])
    with pytest.raises(FusionError, match="rows"):
        fuse(
            grid,
            posteriors=[[1.0]],
            config=FusionConfig(policy="posterior_weighted"),
        )


def test_posterior_wrong_column_count_raises() -> None:
    """Posteriors with wrong row length raises FusionError."""
    grid = progressive_align([("a", "b"), ("c", "d")])
    with pytest.raises(FusionError, match="length"):
        fuse(
            grid,
            posteriors=[[1.0], [1.0, 2.0]],
            config=FusionConfig(policy="posterior_weighted"),
        )


def test_posterior_multi_column() -> None:
    """Posterior weighting applied independently per column."""
    grid = progressive_align([("a", "x"), ("b", "x"), ("b", "y")])
    result = fuse(
        grid,
        posteriors=[[0.9, 0.1], [0.05, 0.8], [0.05, 0.1]],
        config=FusionConfig(policy="posterior_weighted"),
    )
    # Col 0: "a"=0.9, "b"=0.05+0.05=0.1 => "a" wins
    assert result.tokens[0] == "a"
    # Col 1: "x"=0.1+0.8=0.9, "y"=0.1 => "x" wins
    assert result.tokens[1] == "x"
