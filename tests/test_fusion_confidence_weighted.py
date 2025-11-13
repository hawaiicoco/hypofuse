"""Confidence-weighted voting policy tests."""

from __future__ import annotations

import random

import pytest

from hypofuse.exceptions import FusionError
from hypofuse.fusion import FusionConfig, fuse
from hypofuse.multi_align import progressive_align


def test_zero_weight_hypothesis_cannot_decide_tie() -> None:
    """A hypothesis with weight 0 is effectively silenced."""
    # hyp0="a" weight=0, hyp1="b" weight=1, hyp2="b" weight=1
    # "b" gets 2*1=2, "a" gets 1*0=0 => "b" wins
    grid = progressive_align([("a",), ("b",), ("b",)])
    result = fuse(
        grid,
        weights=[0.0, 1.0, 1.0],
        config=FusionConfig(policy="confidence_weighted"),
    )
    assert result.tokens[0] == "b"


def test_zero_weight_breaks_tie_against_silenced() -> None:
    """Two-way tie where one side has weight 0: the other side wins."""
    grid = progressive_align([("a",), ("b",)])
    result = fuse(
        grid,
        weights=[0.0, 1.0],
        config=FusionConfig(policy="confidence_weighted"),
    )
    assert result.tokens[0] == "b"


def test_uniform_weights_reproduce_majority() -> None:
    """Uniform confidence weights give the same result as plain majority."""
    grid = progressive_align([("the", "cat"), ("a", "cat"), ("the", "dog")])
    majority = fuse(grid, config=FusionConfig(policy="majority"))
    weighted = fuse(
        grid,
        weights=[1.0, 1.0, 1.0],
        config=FusionConfig(policy="confidence_weighted"),
    )
    assert weighted.tokens == majority.tokens


def test_uniform_weights_property_over_seeded_grids() -> None:
    """Property test: uniform weights == majority for random grids."""
    rng = random.Random(123)
    vocab = ["alpha", "beta", "gamma", "delta"]
    for _ in range(50):
        n_hyps = rng.randint(2, 4)
        hyps = []
        for _ in range(n_hyps):
            length = rng.randint(1, 3)
            hyps.append(tuple(rng.choice(vocab) for _ in range(length)))
        grid = progressive_align(hyps)
        majority = fuse(grid, config=FusionConfig(policy="majority"))
        weighted = fuse(
            grid,
            weights=[1.0] * n_hyps,
            config=FusionConfig(policy="confidence_weighted"),
        )
        assert weighted.tokens == majority.tokens


def test_wrong_weight_count_raises() -> None:
    """weights length must match grid depth."""
    grid = progressive_align([("a",), ("b",)])
    with pytest.raises(FusionError, match="weights"):
        fuse(
            grid,
            weights=[1.0],
            config=FusionConfig(policy="confidence_weighted"),
        )


def test_default_weights_are_uniform() -> None:
    """Omitting weights defaults to uniform (all 1.0), matching majority."""
    grid = progressive_align([("x",), ("y",), ("x",)])
    majority = fuse(grid, config=FusionConfig(policy="majority"))
    weighted = fuse(grid, config=FusionConfig(policy="confidence_weighted"))
    assert weighted.tokens == majority.tokens
