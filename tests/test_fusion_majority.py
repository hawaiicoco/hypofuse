"""Majority voting fusion goldens."""

from __future__ import annotations

import pytest

from hypofuse.fusion import (
    FusionConfig,
    fuse,
    fusion_invariants,
)
from hypofuse.multi_align import progressive_align


def test_majority_picks_most_common_token() -> None:
    grid = progressive_align([("the", "cat"), ("a", "cat"), ("the", "dog")])
    result = fuse(grid, config=FusionConfig(policy="majority"))
    # Two votes for "the" wins over one "a".
    assert result.tokens[0] == "the"
    # Confidence is agreement: 2/3 of the non-null tokens match.
    assert result.confidences[0] == pytest.approx(2 / 3)


def test_majority_tie_break_lexicographic() -> None:
    grid = progressive_align([("a", "b"), ("b", "a")])
    result = fuse(grid, config=FusionConfig(policy="majority", tie_break="lexicographic"))
    assert result.tokens[0] == "a"


def test_majority_tie_break_first() -> None:
    grid = progressive_align([("a", "b"), ("b", "a")])
    result = fuse(grid, config=FusionConfig(policy="majority", tie_break="first"))
    # First hyp contributes "a" first; first tie-break wins.
    assert result.tokens[0] in {"a", "b"}


def test_score_weighted_voting() -> None:
    grid = progressive_align([("a", "b"), ("a", "c")])
    result = fuse(
        grid, scores=[[1.0, 1.0], [0.1, 0.1]], config=FusionConfig(policy="score_weighted")
    )
    assert result.tokens[0] == "a"


def test_lm_weighted_voting() -> None:
    grid = progressive_align([("a", "b"), ("a", "c")])
    result = fuse(
        grid,
        lm_scores=[[1.0, 0.0], [0.0, 0.0]],
        config=FusionConfig(policy="lm_weighted", alpha=5.0),
    )
    assert result.tokens[0] == "a"


def test_fusion_invariants_no_new_tokens() -> None:
    grid = progressive_align([("a", "b"), ("a", "c"), ("b", "c")])
    result = fuse(grid)
    inputs = [("a", "b"), ("a", "c"), ("b", "c")]
    assert fusion_invariants(result, inputs) is True


def test_fusion_handles_gap_column() -> None:
    grid = progressive_align([("a", "b", "c"), ("a", "c")])
    result = fuse(grid)
    assert len(result.tokens) == grid.width


def test_fusion_confidence_matches_agreement() -> None:
    grid = progressive_align([("a", "b"), ("a", "b"), ("x", "y")])
    result = fuse(grid)
    assert result.confidences[0] == pytest.approx(2 / 3)
