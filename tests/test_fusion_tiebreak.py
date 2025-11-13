"""Tie-break extension tests: highest_score and seeded."""

from __future__ import annotations

import pytest

from hypofuse.exceptions import FusionError
from hypofuse.fusion import FusionConfig, fuse
from hypofuse.multi_align import progressive_align


def test_highest_score_picks_better_hypothesis_token() -> None:
    """Among tied tokens, highest_score picks the one from the better-scoring hyp."""
    # Grid: ("a", "b") where each hyp contributes one token => tie.
    # Scores: hyp0 has score 0.9, hyp1 has score 0.1.
    # "a" has max weight 0.9, "b" has max weight 0.1 => "a" wins.
    grid = progressive_align([("a",), ("b",)])
    result = fuse(
        grid,
        scores=[[0.9], [0.1]],
        config=FusionConfig(policy="score_weighted", tie_break="highest_score"),
    )
    assert result.tokens[0] == "a"


def test_highest_score_reversed_scores() -> None:
    """Reversing scores reverses the highest_score tie-break winner."""
    grid = progressive_align([("a",), ("b",)])
    result = fuse(
        grid,
        scores=[[0.1], [0.9]],
        config=FusionConfig(policy="score_weighted", tie_break="highest_score"),
    )
    assert result.tokens[0] == "b"


def test_seeded_is_reproducible() -> None:
    """Fixed seed produces the same result across calls."""
    grid = progressive_align([("a",), ("b",), ("c",)])
    cfg = FusionConfig(tie_break="seeded", seed=42)
    r1 = fuse(grid, config=cfg)
    r2 = fuse(grid, config=cfg)
    assert r1.tokens == r2.tokens


def test_seeded_can_differ_across_seeds() -> None:
    """Different seeds can produce different winners (not always, but possible)."""
    grid = progressive_align([("a",), ("b",), ("c",), ("d",)])
    results = set()
    for seed in range(20):
        cfg = FusionConfig(tie_break="seeded", seed=seed)
        r = fuse(grid, config=cfg)
        results.add(r.tokens[0])
    # With 4 candidates and 20 seeds, we should see at least 2 different winners.
    assert len(results) >= 2


def test_unknown_tie_break_still_raises() -> None:
    with pytest.raises(FusionError, match="tie_break"):
        FusionConfig(tie_break="random").validate()


def test_highest_score_with_majority_no_tie() -> None:
    """When there is a clear majority, highest_score does not change the winner."""
    grid = progressive_align([("a", "b"), ("a", "c"), ("x", "y")])
    result = fuse(
        grid,
        scores=[[1.0, 1.0], [0.5, 0.5], [0.1, 0.1]],
        config=FusionConfig(policy="score_weighted", tie_break="highest_score"),
    )
    assert result.tokens[0] == "a"


def test_seeded_default_seed_is_zero() -> None:
    """Default seed=0 produces a deterministic result."""
    grid = progressive_align([("a",), ("b",)])
    cfg1 = FusionConfig(tie_break="seeded")
    cfg2 = FusionConfig(tie_break="seeded", seed=0)
    assert fuse(grid, config=cfg1).tokens == fuse(grid, config=cfg2).tokens
